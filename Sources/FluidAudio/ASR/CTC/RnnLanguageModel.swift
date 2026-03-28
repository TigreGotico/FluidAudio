/// CoreML RNN language model for BPE-level beam search rescoring.
///
/// Wraps a step-by-step LSTM LM that takes one token at a time and
/// maintains hidden state. Used by both RNNT and CTC beam search decoders
/// for token-level scoring (better than word-level ARPA).
///
/// The CoreML model expects:
///   - Inputs:  token_id [1] int32, h_in [layers, 1, hidden] f32, c_in [layers, 1, hidden] f32
///   - Outputs: log_probs [1, vocab] f32, h_out [layers, 1, hidden] f32, c_out [layers, 1, hidden] f32

import CoreML
import Foundation

public struct RnnLanguageModel {

    private let model: MLModel
    public let vocabSize: Int
    public let numLayers: Int
    public let hiddenDim: Int
    private let predictionOptions: MLPredictionOptions

    public init(model: MLModel, vocabSize: Int, numLayers: Int, hiddenDim: Int) {
        self.model = model
        self.vocabSize = vocabSize
        self.numLayers = numLayers
        self.hiddenDim = hiddenDim
        self.predictionOptions = MLPredictionOptions()
    }

    /// Load from a compiled .mlmodelc or .mlpackage directory.
    public static func load(
        from url: URL,
        vocabSize: Int,
        numLayers: Int,
        hiddenDim: Int,
        computeUnits: MLComputeUnits = .all
    ) throws -> RnnLanguageModel {
        let fm = FileManager.default
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        let compiledURL = url.appendingPathExtension("mlmodelc")
        let packageURL = url.pathExtension == "mlmodelc" ? url
            : url.pathExtension == "mlpackage" ? url : compiledURL

        let mlModel: MLModel
        if fm.fileExists(atPath: url.path), url.pathExtension == "mlmodelc" {
            mlModel = try MLModel(contentsOf: url, configuration: config)
        } else if fm.fileExists(atPath: url.path), url.pathExtension == "mlpackage" {
            let compiled = try MLModel.compileModel(at: url)
            mlModel = try MLModel(contentsOf: compiled, configuration: config)
        } else {
            // Try appending extensions
            let mlmodelc = url.appendingPathExtension("mlmodelc")
            let mlpackage = url.appendingPathExtension("mlpackage")
            if fm.fileExists(atPath: mlmodelc.path) {
                mlModel = try MLModel(contentsOf: mlmodelc, configuration: config)
            } else if fm.fileExists(atPath: mlpackage.path) {
                let compiled = try MLModel.compileModel(at: mlpackage)
                mlModel = try MLModel(contentsOf: compiled, configuration: config)
            } else {
                throw RnnLmError.modelNotFound(url.path)
            }
        }

        return RnnLanguageModel(model: mlModel, vocabSize: vocabSize,
                                numLayers: numLayers, hiddenDim: hiddenDim)
    }

    /// Create zero-initialized LSTM state.
    public func makeInitialState() throws -> (h: MLMultiArray, c: MLMultiArray) {
        let shape = [numLayers, 1, hiddenDim] as [NSNumber]
        let h = try MLMultiArray(shape: shape, dataType: .float32)
        let c = try MLMultiArray(shape: shape, dataType: .float32)
        let count = numLayers * hiddenDim
        memset(h.dataPointer, 0, count * 4)
        memset(c.dataPointer, 0, count * 4)
        return (h, c)
    }

    /// Score a single token given LSTM state. Returns log_probs pointer and new state.
    ///
    /// This is the core method used in beam search. Each beam hypothesis carries
    /// its own (h, c) state, so this method does not mutate any shared state.
    ///
    /// - Parameters:
    ///   - tokenId: BPE token ID to score
    ///   - h: LSTM hidden state [numLayers, 1, hiddenDim]
    ///   - c: LSTM cell state [numLayers, 1, hiddenDim]
    /// - Returns: (logProbs MLMultiArray [1, vocabSize], h_out, c_out)
    public func score(
        tokenId: Int, h: MLMultiArray, c: MLMultiArray
    ) throws -> (logProbs: MLMultiArray, hOut: MLMultiArray, cOut: MLMultiArray) {
        let tokenArray = try MLMultiArray(shape: [1], dataType: .int32)
        tokenArray[0] = NSNumber(value: Int32(tokenId))

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "token_id": MLFeatureValue(multiArray: tokenArray),
            "h_in": MLFeatureValue(multiArray: h),
            "c_in": MLFeatureValue(multiArray: c),
        ])

        let output = try model.prediction(from: input, options: predictionOptions)

        let logProbs = output.featureValue(for: "log_probs")!.multiArrayValue!
        let hOut = output.featureValue(for: "h_out")!.multiArrayValue!
        let cOut = output.featureValue(for: "c_out")!.multiArrayValue!

        return (logProbs, hOut, cOut)
    }
}

public enum RnnLmError: Error, LocalizedError {
    case modelNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path):
            return "RNN-LM model not found at: \(path) (tried .mlmodelc and .mlpackage)"
        }
    }
}
