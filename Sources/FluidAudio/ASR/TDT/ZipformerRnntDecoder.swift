/// Greedy RNN-T decoder for Zipformer2 transducer models (icefall/sherpa-onnx).
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

internal struct ZipformerRnntDecoder {

    private let logger = AppLogger(category: "ZipformerRNNT")
    private let config: ASRConfig
    private let predictionOptions = AsrModels.optimizedPredictionOptions()

    init(config: ASRConfig) {
        self.config = config
    }

    /// Decode encoder output using greedy RNNT search.
    ///
    /// The encoder output shape is `[1, T, joinerDim]` (time-major, unlike Parakeet's `[1, D, T]`).
    ///
    /// - Parameters:
    ///   - encoderOutput: Encoder output, shape `[1, T, joinerDim]`
    ///   - encoderSequenceLength: Number of valid encoder frames
    ///   - decoderModel: Stateless decoder CoreML model
    ///   - joinerModel: Joiner CoreML model
    ///   - blankId: Blank token ID (typically 0 for Zipformer2)
    ///   - contextSize: Decoder context window size (typically 2)
    /// - Returns: Decoded hypothesis with tokens, timestamps, and confidences
    func decode(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        decoderModel: MLModel,
        joinerModel: MLModel,
        blankId: Int,
        contextSize: Int
    ) throws -> TdtHypothesis {
        let joinerDim = encoderOutput.shape[2].intValue

        // Context buffer: last `contextSize` tokens, initialized with blank
        var context = [Int](repeating: blankId, count: contextSize)

        // Use TdtHypothesis for compatibility with existing pipeline
        // We create a dummy decoder state since Zipformer2 is stateless
        var hypothesis = TdtDecoderState.make(decoderLayers: 1)
        var result = TdtHypothesis(decState: hypothesis)

        // Precompute encoder strides for efficient frame extraction
        // Shape: [1, T, joinerDim]
        let encStride0 = encoderOutput.strides[0].intValue
        let encStride1 = encoderOutput.strides[1].intValue
        let encStride2 = encoderOutput.strides[2].intValue
        let encPtr = encoderOutput.dataPointer.bindMemory(
            to: Float.self, capacity: encoderOutput.count)

        // Preallocate reusable arrays
        let encoderStep = try MLMultiArray(
            shape: [1, NSNumber(value: joinerDim)], dataType: .float32)
        let decoderInput = try MLMultiArray(
            shape: [1, NSNumber(value: contextSize)], dataType: .int32)

        let encStepPtr = encoderStep.dataPointer.bindMemory(
            to: Float.self, capacity: joinerDim)

        for t in 0..<encoderSequenceLength {
            // Extract encoder frame: encoderOutput[0, t, :] -> [1, joinerDim]
            for d in 0..<joinerDim {
                encStepPtr[d] = encPtr[0 * encStride0 + t * encStride1 + d * encStride2]
            }

            // One prediction per encoder frame (matches Python reference decoder).
            // Run stateless decoder with context tokens
            for i in 0..<contextSize {
                decoderInput[i] = NSNumber(value: Int32(context[i]))
            }

            let decInput = try MLDictionaryFeatureProvider(dictionary: [
                "y": MLFeatureValue(multiArray: decoderInput)
            ])
            let decOutput = try decoderModel.prediction(
                from: decInput, options: predictionOptions)
            let decoderOut = decOutput.featureValue(for: "decoder_out")!.multiArrayValue!

            // Run joiner: encoder_out + decoder_out -> logits
            let joinInput = try MLDictionaryFeatureProvider(dictionary: [
                "encoder_out": MLFeatureValue(multiArray: encoderStep),
                "decoder_out": MLFeatureValue(multiArray: decoderOut),
            ])
            let joinOutput = try joinerModel.prediction(
                from: joinInput, options: predictionOptions)
            let logits = joinOutput.featureValue(for: "logit")!.multiArrayValue!

            // Argmax over vocabulary using vDSP
            let vocabSize = logits.shape.last!.intValue
            let logitsPtr = logits.dataPointer.bindMemory(
                to: Float.self, capacity: vocabSize)
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(logitsPtr, 1, &maxVal, &maxIdx, vDSP_Length(vocabSize))
            let tokenId = Int(maxIdx)

            if tokenId != blankId {
                // Emit token
                result.ySequence.append(tokenId)
                result.timestamps.append(t)
                result.tokenDurations.append(1)
                result.tokenConfidences.append(maxVal)
                result.lastToken = tokenId

                // Update context: shift left, add new token
                context.removeFirst()
                context.append(tokenId)
            }
        }

        return result
    }
}
