/// RNN-T decoder for Zipformer2 transducer models (icefall/sherpa-onnx).
///
/// Supports both greedy and modified beam search decoding, with optional
/// ARPA language model rescoring at word boundaries.
///
/// Unlike Parakeet TDT, Zipformer2 uses:
/// - **Stateless decoder**: context window of token IDs (no LSTM hidden/cell states)
/// - **Standard RNNT**: no duration prediction, advance one encoder frame per step
/// - **blank_id = 0**: first token in vocabulary is blank
///
/// The decoder takes the last `contextSize` token IDs as input and produces
/// a decoder embedding. The joiner combines encoder + decoder embeddings to
/// produce logits over the vocabulary.

import Accelerate
import CoreML
import Foundation
import OSLog

// MARK: - Beam hypothesis for RNNT modified beam search

internal struct RnntBeam {
    var tokens: [Int]
    var context: [Int]
    var logProb: Float
    var lmScore: Float
    var wordPieces: [String]
    var prevWord: String?
    var timestamps: [Int]
    var confidences: [Float]

    var total: Float { logProb + lmScore }
}

// MARK: - Decoder

internal struct ZipformerRnntDecoder {

    private let logger = AppLogger(category: "ZipformerRNNT")
    private let config: ASRConfig
    private let predictionOptions = AsrModels.optimizedPredictionOptions()

    init(config: ASRConfig) {
        self.config = config
    }

    // MARK: - Greedy decode (one token per frame)

    /// Decode encoder output using greedy RNNT search.
    func decode(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        decoderModel: MLModel,
        joinerModel: MLModel,
        blankId: Int,
        contextSize: Int
    ) throws -> TdtHypothesis {
        let joinerDim = encoderOutput.shape[2].intValue

        var context = [Int](repeating: blankId, count: contextSize)
        let hypothesis = TdtDecoderState.make(decoderLayers: 1)
        var result = TdtHypothesis(decState: hypothesis)

        let encStride0 = encoderOutput.strides[0].intValue
        let encStride1 = encoderOutput.strides[1].intValue
        let encStride2 = encoderOutput.strides[2].intValue
        let encPtr = encoderOutput.dataPointer.bindMemory(
            to: Float.self, capacity: encoderOutput.count)

        let encoderStep = try MLMultiArray(
            shape: [1, NSNumber(value: joinerDim)], dataType: .float32)
        let decoderInput = try MLMultiArray(
            shape: [1, NSNumber(value: contextSize)], dataType: .int32)
        let encStepPtr = encoderStep.dataPointer.bindMemory(
            to: Float.self, capacity: joinerDim)

        for t in 0..<encoderSequenceLength {
            for d in 0..<joinerDim {
                encStepPtr[d] = encPtr[0 * encStride0 + t * encStride1 + d * encStride2]
            }

            for i in 0..<contextSize {
                decoderInput[i] = NSNumber(value: Int32(context[i]))
            }

            let decInput = try MLDictionaryFeatureProvider(dictionary: [
                "y": MLFeatureValue(multiArray: decoderInput)
            ])
            let decOutput = try decoderModel.prediction(
                from: decInput, options: predictionOptions)
            let decoderOut = decOutput.featureValue(for: "decoder_out")!.multiArrayValue!

            let joinInput = try MLDictionaryFeatureProvider(dictionary: [
                "encoder_out": MLFeatureValue(multiArray: encoderStep),
                "decoder_out": MLFeatureValue(multiArray: decoderOut),
            ])
            let joinOutput = try joinerModel.prediction(
                from: joinInput, options: predictionOptions)
            let logits = joinOutput.featureValue(for: "logit")!.multiArrayValue!

            let vocabSize = logits.shape.last!.intValue
            let logitsPtr = logits.dataPointer.bindMemory(
                to: Float.self, capacity: vocabSize)
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(logitsPtr, 1, &maxVal, &maxIdx, vDSP_Length(vocabSize))
            let tokenId = Int(maxIdx)

            if tokenId != blankId {
                result.ySequence.append(tokenId)
                result.timestamps.append(t)
                result.tokenDurations.append(1)
                result.tokenConfidences.append(maxVal)
                result.lastToken = tokenId

                context.removeFirst()
                context.append(tokenId)
            }
        }

        return result
    }

    // MARK: - Modified beam search with optional LM

    /// Decode encoder output using modified beam search with optional ARPA LM.
    ///
    /// Maintains `beamWidth` hypotheses. At each encoder frame, expands each
    /// hypothesis by trying blank + top-K non-blank tokens, then prunes to
    /// the best `beamWidth` hypotheses by score.
    ///
    /// Word-level LM scores are applied at SentencePiece word boundaries (▁ prefix).
    ///
    /// - Parameters:
    ///   - encoderOutput: Encoder output, shape `[1, T, joinerDim]`
    ///   - encoderSequenceLength: Number of valid encoder frames
    ///   - decoderModel: Stateless decoder CoreML model
    ///   - joinerModel: Joiner CoreML model
    ///   - vocabulary: Token ID → string mapping for LM word boundary detection
    ///   - lm: Optional ARPA language model
    ///   - blankId: Blank token ID (typically 0)
    ///   - contextSize: Decoder context window size (typically 2)
    ///   - beamWidth: Number of hypotheses to maintain (default 4)
    ///   - lmWeight: LM score scaling factor (default 0.3)
    ///   - tokenCandidates: Top-K non-blank tokens to consider per frame (default 8)
    func beamDecode(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        decoderModel: MLModel,
        joinerModel: MLModel,
        vocabulary: [Int: String],
        lm: ARPALanguageModel?,
        blankId: Int,
        contextSize: Int,
        beamWidth: Int = 4,
        lmWeight: Float = 0.3,
        tokenCandidates: Int = 8
    ) throws -> TdtHypothesis {
        let joinerDim = encoderOutput.shape[2].intValue

        let encStride0 = encoderOutput.strides[0].intValue
        let encStride1 = encoderOutput.strides[1].intValue
        let encStride2 = encoderOutput.strides[2].intValue
        let encPtr = encoderOutput.dataPointer.bindMemory(
            to: Float.self, capacity: encoderOutput.count)

        let encoderStep = try MLMultiArray(
            shape: [1, NSNumber(value: joinerDim)], dataType: .float32)
        let decoderInput = try MLMultiArray(
            shape: [1, NSNumber(value: contextSize)], dataType: .int32)
        let encStepPtr = encoderStep.dataPointer.bindMemory(
            to: Float.self, capacity: joinerDim)

        // Initialize with single blank-context beam
        let initialContext = [Int](repeating: blankId, count: contextSize)
        var beams = [RnntBeam(
            tokens: [], context: initialContext, logProb: 0.0, lmScore: 0.0,
            wordPieces: [], prevWord: nil, timestamps: [], confidences: []
        )]

        // Cache decoder outputs for each unique context to avoid redundant calls
        var decoderCache: [[Int]: MLMultiArray] = [:]

        for t in 0..<encoderSequenceLength {
            // Extract encoder frame
            for d in 0..<joinerDim {
                encStepPtr[d] = encPtr[0 * encStride0 + t * encStride1 + d * encStride2]
            }

            var candidates: [RnntBeam] = []
            candidates.reserveCapacity(beams.count * (tokenCandidates + 1))

            for beam in beams {
                // Get decoder output (with caching)
                let decoderOut: MLMultiArray
                if let cached = decoderCache[beam.context] {
                    decoderOut = cached
                } else {
                    for i in 0..<contextSize {
                        decoderInput[i] = NSNumber(value: Int32(beam.context[i]))
                    }
                    let decInput = try MLDictionaryFeatureProvider(dictionary: [
                        "y": MLFeatureValue(multiArray: decoderInput)
                    ])
                    let decOutput = try decoderModel.prediction(
                        from: decInput, options: predictionOptions)
                    decoderOut = decOutput.featureValue(for: "decoder_out")!.multiArrayValue!
                    decoderCache[beam.context] = decoderOut
                }

                // Run joiner
                let joinInput = try MLDictionaryFeatureProvider(dictionary: [
                    "encoder_out": MLFeatureValue(multiArray: encoderStep),
                    "decoder_out": MLFeatureValue(multiArray: decoderOut),
                ])
                let joinOutput = try joinerModel.prediction(
                    from: joinInput, options: predictionOptions)
                let logits = joinOutput.featureValue(for: "logit")!.multiArrayValue!

                let vs = logits.shape.last!.intValue
                let logitsPtr = logits.dataPointer.bindMemory(to: Float.self, capacity: vs)

                // Compute log-softmax in place
                var logProbs = [Float](repeating: 0, count: vs)
                var maxLogit: Float = -Float.infinity
                for v in 0..<vs {
                    if logitsPtr[v] > maxLogit { maxLogit = logitsPtr[v] }
                }
                var sumExp: Float = 0
                for v in 0..<vs {
                    sumExp += exp(logitsPtr[v] - maxLogit)
                }
                let logSumExp = maxLogit + log(sumExp)
                for v in 0..<vs {
                    logProbs[v] = logitsPtr[v] - logSumExp
                }

                // Candidate 1: blank (stay on same beam, advance frame)
                candidates.append(RnntBeam(
                    tokens: beam.tokens, context: beam.context,
                    logProb: beam.logProb + logProbs[blankId],
                    lmScore: beam.lmScore,
                    wordPieces: beam.wordPieces, prevWord: beam.prevWord,
                    timestamps: beam.timestamps, confidences: beam.confidences
                ))

                // Find top-K non-blank tokens
                var indexed = [(Int, Float)]()
                indexed.reserveCapacity(vs)
                for v in 0..<vs where v != blankId {
                    indexed.append((v, logProbs[v]))
                }
                indexed.sort { $0.1 > $1.1 }

                // Candidate 2..K+1: top non-blank tokens
                for (tokenId, tokenLogProb) in indexed.prefix(tokenCandidates) {
                    var newContext = beam.context
                    newContext.removeFirst()
                    newContext.append(tokenId)

                    // LM scoring at word boundaries
                    var newLmScore = beam.lmScore
                    var newWordPieces = beam.wordPieces
                    var newPrevWord = beam.prevWord

                    if let lm = lm, let tokenStr = vocabulary[tokenId] {
                        newWordPieces.append(tokenStr)
                        // Check for word boundary: SentencePiece ▁ prefix on NEXT token
                        if tokenStr.hasPrefix("\u{2581}") && !beam.wordPieces.isEmpty {
                            // Previous word pieces form a complete word
                            let word = beam.wordPieces.joined()
                                .replacingOccurrences(of: "\u{2581}", with: "")
                            if !word.isEmpty {
                                newLmScore += lmWeight * lm.score(
                                    word: word.lowercased(), prev: beam.prevWord)
                                newPrevWord = word.lowercased()
                            }
                            newWordPieces = [tokenStr]
                        }
                    }

                    var newTimestamps = beam.timestamps
                    newTimestamps.append(t)
                    var newConfidences = beam.confidences
                    newConfidences.append(exp(tokenLogProb))

                    candidates.append(RnntBeam(
                        tokens: beam.tokens + [tokenId],
                        context: newContext,
                        logProb: beam.logProb + tokenLogProb,
                        lmScore: newLmScore,
                        wordPieces: newWordPieces,
                        prevWord: newPrevWord,
                        timestamps: newTimestamps,
                        confidences: newConfidences
                    ))
                }
            }

            // Prune to top beamWidth by total score
            candidates.sort { $0.total > $1.total }
            beams = Array(candidates.prefix(beamWidth))

            // Clear decoder cache for contexts no longer in active beams
            let activeContexts = Set(beams.map { $0.context })
            decoderCache = decoderCache.filter { activeContexts.contains($0.key) }
        }

        // Score final incomplete word for LM
        if let lm = lm {
            for i in 0..<beams.count {
                let word = beams[i].wordPieces.joined()
                    .replacingOccurrences(of: "\u{2581}", with: "")
                if !word.isEmpty {
                    beams[i].lmScore += lmWeight * lm.score(
                        word: word.lowercased(), prev: beams[i].prevWord)
                }
            }
        }

        // Select best beam
        let best = beams.max(by: { $0.total < $1.total }) ?? beams[0]

        // Convert to TdtHypothesis
        let hypothesis = TdtDecoderState.make(decoderLayers: 1)
        var result = TdtHypothesis(decState: hypothesis)
        result.ySequence = best.tokens
        result.timestamps = best.timestamps
        result.tokenConfidences = best.confidences
        result.tokenDurations = [Int](repeating: 1, count: best.tokens.count)
        result.lastToken = best.tokens.last ?? blankId

        return result
    }
}
