@preconcurrency import CoreML
import Foundation
import OSLog

extension AsrManager {

    internal func transcribeWithState(
        _ audioSamples: [Float], source: AudioSource
    ) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        guard audioSamples.count >= 16_000 else { throw ASRError.invalidAudioData }

        let startTime = Date()

        // Get the appropriate decoder state
        var decoderState: TdtDecoderState
        switch source {
        case .microphone:
            decoderState = microphoneDecoderState
        case .system:
            decoderState = systemDecoderState
        }

        // Route to appropriate processing method based on audio length
        let maxSamples = getMaxModelSamples()
        if audioSamples.count <= maxSamples {
            let originalLength = audioSamples.count
            let frameAlignedCandidate =
                ((originalLength + ASRConstants.samplesPerEncoderFrame - 1)
                    / ASRConstants.samplesPerEncoderFrame) * ASRConstants.samplesPerEncoderFrame
            let frameAlignedLength: Int
            let alignedSamples: [Float]
            if frameAlignedCandidate > originalLength && frameAlignedCandidate <= maxSamples {
                frameAlignedLength = frameAlignedCandidate
                alignedSamples = audioSamples + Array(repeating: 0, count: frameAlignedLength - originalLength)
            } else {
                frameAlignedLength = originalLength
                alignedSamples = audioSamples
            }
            let paddedAudio: [Float] = padAudioIfNeeded(alignedSamples, targetLength: maxSamples)
            let (hypothesis, encoderSequenceLength) = try await executeMLInferenceWithTimings(
                paddedAudio,
                originalLength: frameAlignedLength,
                actualAudioFrames: nil,  // Will be calculated from originalLength
                decoderState: &decoderState
            )

            var result = processTranscriptionResult(
                tokenIds: hypothesis.ySequence,
                timestamps: hypothesis.timestamps,
                confidences: hypothesis.tokenConfidences,
                tokenDurations: hypothesis.tokenDurations,
                encoderSequenceLength: encoderSequenceLength,
                audioSamples: audioSamples,
                processingTime: Date().timeIntervalSince(startTime)
            )

            // Auto-apply vocabulary rescoring when configured
            if vocabBoostingEnabled {
                result = await applyVocabularyRescoring(result: result, audioSamples: audioSamples)
            }

            // Store decoder state back
            switch source {
            case .microphone:
                microphoneDecoderState = decoderState
            case .system:
                systemDecoderState = decoderState
            }

            return result
        }

        // ChunkProcessor handles stateless chunked transcription for long audio
        let processor = ChunkProcessor(audioSamples: audioSamples)
        var result = try await processor.process(
            using: self,
            startTime: startTime,
            progressHandler: { [weak self] progress in
                guard let self else { return }
                await self.progressEmitter.report(progress: progress)
            }
        )

        // Auto-apply vocabulary rescoring when configured
        if vocabBoostingEnabled {
            result = await applyVocabularyRescoring(result: result, audioSamples: audioSamples)
        }

        // Store decoder state back (ChunkProcessor uses the stored state directly)
        switch source {
        case .microphone:
            microphoneDecoderState = decoderState
        case .system:
            systemDecoderState = decoderState
        }

        return result
    }

    internal func executeMLInferenceWithTimings(
        _ paddedAudio: [Float],
        originalLength: Int? = nil,
        actualAudioFrames: Int? = nil,
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int = 0,
        isLastChunk: Bool = false,
        globalFrameOffset: Int = 0
    ) async throws -> (hypothesis: TdtHypothesis, encoderSequenceLength: Int) {

        // Zipformer2 with separate mel path (non-fused): compute mel → run encoder → decode
        // For fused Zipformer2, fall through to the normal Parakeet path below (same audio_signal interface)
        if let models = asrModels, models.version.requiresMelInput, !models.hasFusedMel {
            return try await executeZipformerInference(
                paddedAudio,
                originalLength: originalLength,
                actualAudioFrames: actualAudioFrames,
                decoderState: &decoderState,
                contextFrameAdjustment: contextFrameAdjustment,
                isLastChunk: isLastChunk,
                globalFrameOffset: globalFrameOffset
            )
        }

        let preprocessorInput = try await preparePreprocessorInput(
            paddedAudio, actualLength: originalLength)

        let preprocessorAudioArray = preprocessorInput.featureValue(for: "audio_signal")?.multiArrayValue

        do {
            guard let preprocessorModel = preprocessorModel else {
                throw ASRError.notInitialized
            }

            try Task.checkCancellation()
            let preprocessorOutput = try await preprocessorModel.compatPrediction(
                from: preprocessorInput,
                options: predictionOptions
            )

            let encoderOutputProvider: MLFeatureProvider
            if let encoderModel = encoderModel {
                // Split frontend: run separate encoder
                let encoderInput = try prepareEncoderInput(
                    encoder: encoderModel,
                    preprocessorOutput: preprocessorOutput,
                    originalInput: preprocessorInput
                )

                try Task.checkCancellation()
                encoderOutputProvider = try await encoderModel.compatPrediction(
                    from: encoderInput,
                    options: predictionOptions
                )
            } else {
                // Fused frontend: preprocessor output already contains encoder features
                encoderOutputProvider = preprocessorOutput
            }

            // Parakeet uses "encoder"/"encoder_length", Zipformer2 fused uses "encoder_out"/"encoder_out_lens"
            let encoderKey = encoderOutputProvider.featureValue(for: "encoder") != nil ? "encoder" : "encoder_out"
            let lengthKey =
                encoderOutputProvider.featureValue(for: "encoder_length") != nil
                ? "encoder_length" : "encoder_out_lens"
            let rawEncoderOutput = try extractFeatureValue(
                from: encoderOutputProvider, key: encoderKey, errorMessage: "Invalid encoder output")
            let encoderLength = try extractFeatureValue(
                from: encoderOutputProvider, key: lengthKey,
                errorMessage: "Invalid encoder output length")

            let encoderSequenceLength = encoderLength[0].intValue

            // Calculate actual audio frames if not provided using shared constants
            let actualFrames =
                actualAudioFrames ?? ASRConstants.calculateEncoderFrames(from: originalLength ?? paddedAudio.count)

            let hypothesis = try await tdtDecodeWithTimings(
                encoderOutput: rawEncoderOutput,
                encoderSequenceLength: encoderSequenceLength,
                actualAudioFrames: actualFrames,
                originalAudioSamples: paddedAudio,
                decoderState: &decoderState,
                contextFrameAdjustment: contextFrameAdjustment,
                isLastChunk: isLastChunk,
                globalFrameOffset: globalFrameOffset
            )

            if let preprocessorAudioArray {
                await sharedMLArrayCache.returnArray(preprocessorAudioArray)
            }

            return (hypothesis, encoderSequenceLength)
        } catch {
            if let preprocessorAudioArray {
                await sharedMLArrayCache.returnArray(preprocessorAudioArray)
            }
            throw error
        }
    }

    private func prepareEncoderInput(
        encoder: MLModel,
        preprocessorOutput: MLFeatureProvider,
        originalInput: MLFeatureProvider
    ) throws -> MLFeatureProvider {
        let inputDescriptions = encoder.modelDescription.inputDescriptionsByName

        let missingNames = inputDescriptions.keys.filter { name in
            preprocessorOutput.featureValue(for: name) == nil
        }

        if missingNames.isEmpty {
            return preprocessorOutput
        }

        var features: [String: MLFeatureValue] = [:]

        for name in inputDescriptions.keys {
            if let value = preprocessorOutput.featureValue(for: name) {
                features[name] = value
                continue
            }

            if let fallback = originalInput.featureValue(for: name) {
                features[name] = fallback
                continue
            }

            let availableInputs = preprocessorOutput.featureNames.sorted().joined(separator: ", ")
            let fallbackInputs = originalInput.featureNames.sorted().joined(separator: ", ")
            throw ASRError.processingFailed(
                "Missing required encoder input: \(name). Available inputs: \(availableInputs), "
                    + "fallback inputs: \(fallbackInputs)"
            )
        }

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    /// Streaming-friendly chunk transcription that preserves decoder state and supports start-frame offset.
    /// This is used by both sliding window chunking and streaming paths to unify behavior.
    public func transcribeStreamingChunk(
        _ chunkSamples: [Float],
        source: AudioSource,
        previousTokens: [Int] = [],
        isLastChunk: Bool = false
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], encoderSequenceLength: Int) {
        // Select and copy decoder state for the source
        var state = (source == .microphone) ? microphoneDecoderState : systemDecoderState
        let maxSamples = getMaxModelSamples()

        let originalLength = chunkSamples.count
        let frameAlignedCandidate =
            ((originalLength + ASRConstants.samplesPerEncoderFrame - 1)
                / ASRConstants.samplesPerEncoderFrame) * ASRConstants.samplesPerEncoderFrame
        let frameAlignedLength: Int
        let alignedSamples: [Float]
        if previousTokens.isEmpty
            && frameAlignedCandidate > originalLength
            && frameAlignedCandidate <= maxSamples
        {
            frameAlignedLength = frameAlignedCandidate
            alignedSamples = chunkSamples + Array(repeating: 0, count: frameAlignedLength - originalLength)
        } else {
            frameAlignedLength = originalLength
            alignedSamples = chunkSamples
        }
        let padded = padAudioIfNeeded(alignedSamples, targetLength: maxSamples)
        let (hypothesis, encLen) = try await executeMLInferenceWithTimings(
            padded,
            originalLength: frameAlignedLength,
            actualAudioFrames: nil,  // Will be calculated from originalLength
            decoderState: &state,
            contextFrameAdjustment: 0,  // Non-streaming chunks don't use adaptive context
            isLastChunk: isLastChunk
        )

        // Persist updated state back to the source-specific slot
        if source == .microphone {
            microphoneDecoderState = state
        } else {
            systemDecoderState = state
        }

        // Apply token deduplication if previous tokens are provided
        if !previousTokens.isEmpty && hypothesis.hasTokens {
            let (deduped, removedCount) = removeDuplicateTokenSequence(
                previous: previousTokens, current: hypothesis.ySequence)
            let adjustedTimestamps =
                removedCount > 0 ? Array(hypothesis.timestamps.dropFirst(removedCount)) : hypothesis.timestamps
            let adjustedConfidences =
                removedCount > 0
                ? Array(hypothesis.tokenConfidences.dropFirst(removedCount)) : hypothesis.tokenConfidences

            return (deduped, adjustedTimestamps, adjustedConfidences, encLen)
        }

        return (hypothesis.ySequence, hypothesis.timestamps, hypothesis.tokenConfidences, encLen)
    }

    internal func processTranscriptionResult(
        tokenIds: [Int],
        timestamps: [Int] = [],
        confidences: [Float] = [],
        tokenDurations: [Int] = [],
        encoderSequenceLength: Int,
        audioSamples: [Float],
        processingTime: TimeInterval,
        tokenTimings: [TokenTiming] = []
    ) -> ASRResult {

        let (text, finalTimings) = convertTokensWithExistingTimings(tokenIds, timings: tokenTimings)
        let duration = TimeInterval(audioSamples.count) / TimeInterval(config.sampleRate)

        // Convert timestamps to TokenTiming objects if provided
        let timingsFromTimestamps = createTokenTimings(
            from: tokenIds, timestamps: timestamps, confidences: confidences, tokenDurations: tokenDurations)

        // Use existing timings if provided, otherwise use timings from timestamps
        let resultTimings = tokenTimings.isEmpty ? timingsFromTimestamps : finalTimings

        // Calculate confidence based on actual model confidence scores from TDT decoder
        let confidence = calculateConfidence(
            duration: duration,
            tokenCount: tokenIds.count,
            isEmpty: text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            tokenConfidences: confidences
        )

        return ASRResult(
            text: text,
            confidence: confidence,
            duration: duration,
            processingTime: processingTime,
            tokenTimings: resultTimings
        )
    }

    nonisolated internal func padAudioIfNeeded(_ audioSamples: [Float], targetLength: Int) -> [Float] {
        guard audioSamples.count < targetLength else { return audioSamples }
        return audioSamples + Array(repeating: 0, count: targetLength - audioSamples.count)
    }

    /// Calculate confidence score based purely on TDT model token confidence scores
    /// Returns the average of token-level softmax probabilities from the decoder
    /// Range: 0.1 (empty transcription) to 1.0 (perfect confidence)
    private func calculateConfidence(
        duration: Double, tokenCount: Int, isEmpty: Bool, tokenConfidences: [Float]
    ) -> Float {
        // Empty transcription gets low confidence
        if isEmpty {
            return 0.1
        }

        // We should always have token confidence scores from the TDT decoder
        guard !tokenConfidences.isEmpty && tokenConfidences.count == tokenCount else {
            logger.warning("Expected token confidences but got none - this should not happen")
            return 0.5  // Default middle confidence if something went wrong
        }

        // Return pure model confidence: average of token-level softmax probabilities
        let meanConfidence = tokenConfidences.reduce(0.0, +) / Float(tokenConfidences.count)

        // Ensure confidence is in valid range (clamp to avoid edge cases)
        return max(0.1, min(1.0, meanConfidence))
    }

    /// Convert frame timestamps to TokenTiming objects
    private func createTokenTimings(
        from tokenIds: [Int], timestamps: [Int], confidences: [Float], tokenDurations: [Int] = []
    ) -> [TokenTiming] {
        guard
            !tokenIds.isEmpty && !timestamps.isEmpty && tokenIds.count == timestamps.count
                && confidences.count == tokenIds.count
        else {
            return []
        }

        var timings: [TokenTiming] = []

        // Create combined data for sorting
        let combinedData = zip(
            zip(zip(tokenIds, timestamps), confidences),
            tokenDurations.isEmpty ? Array(repeating: 0, count: tokenIds.count) : tokenDurations
        ).map {
            (tokenId: $0.0.0.0, timestamp: $0.0.0.1, confidence: $0.0.1, duration: $0.1)
        }

        // Sort by timestamp to ensure chronological order
        let sortedData = combinedData.sorted { $0.timestamp < $1.timestamp }

        for i in 0..<sortedData.count {
            let data = sortedData[i]
            let tokenId = data.tokenId
            let frameIndex = data.timestamp

            // Convert encoder frame index to time (80ms per frame)
            let startTime = TimeInterval(frameIndex) * 0.08

            // Calculate end time using actual token duration if available
            let endTime: TimeInterval
            if !tokenDurations.isEmpty && data.duration > 0 {
                // Use actual token duration (convert frames to time: duration * 0.08)
                let durationInSeconds = TimeInterval(data.duration) * 0.08
                endTime = startTime + max(durationInSeconds, 0.08)  // Minimum 80ms duration
            } else if i < sortedData.count - 1 {
                // Fallback: Use next token's start time as this token's end time
                let nextStartTime = TimeInterval(sortedData[i + 1].timestamp) * 0.08
                endTime = max(nextStartTime, startTime + 0.08)  // Ensure end > start
            } else {
                // Last token: assume minimum duration
                endTime = startTime + 0.08
            }

            // Validate that end time is after start time
            let validatedEndTime = max(endTime, startTime + 0.001)  // Minimum 1ms gap

            // Get token text from vocabulary if available and normalize for timing display
            let rawToken = vocabulary[tokenId] ?? "token_\(tokenId)"
            let tokenText = normalizedTimingToken(rawToken)

            // Use actual confidence score from TDT decoder
            let tokenConfidence = data.confidence

            let timing = TokenTiming(
                token: tokenText,
                tokenId: tokenId,
                startTime: startTime,
                endTime: validatedEndTime,
                confidence: tokenConfidence
            )

            timings.append(timing)
        }
        return timings
    }

    /// Slice encoder output to remove left context frames (following NeMo approach)
    private func sliceEncoderOutput(
        _ encoderOutput: MLMultiArray,
        from startFrame: Int,
        newLength: Int
    ) throws -> MLMultiArray {
        let shape = encoderOutput.shape
        let batchSize = shape[0].intValue
        let hiddenSize = shape[2].intValue

        // Create new array with sliced dimensions
        let slicedArray = try MLMultiArray(
            shape: [batchSize, newLength, hiddenSize] as [NSNumber],
            dataType: encoderOutput.dataType
        )

        // Copy data from startFrame onwards
        let sourcePtr = encoderOutput.dataPointer.bindMemory(to: Float.self, capacity: encoderOutput.count)
        let destPtr = slicedArray.dataPointer.bindMemory(to: Float.self, capacity: slicedArray.count)

        for t in 0..<newLength {
            for h in 0..<hiddenSize {
                let sourceIndex = (startFrame + t) * hiddenSize + h
                let destIndex = t * hiddenSize + h
                destPtr[destIndex] = sourcePtr[sourceIndex]
            }
        }

        return slicedArray
    }

    /// Remove duplicate token sequences at the start of the current list that overlap
    /// with the tail of the previous accumulated tokens. Returns deduplicated current tokens
    /// and the number of removed leading tokens so caller can drop aligned timestamps.
    /// Ideally this is not needed. We need to make some more fixes to the TDT decoding logic,
    /// this should be a temporary workaround.
    internal func removeDuplicateTokenSequence(
        previous: [Int], current: [Int], maxOverlap: Int = 12
    ) -> (deduped: [Int], removedCount: Int) {

        // Handle single punctuation token duplicates first
        let punctuationTokens = [7883, 7952, 7948]  // period, question, exclamation
        var workingCurrent = current
        var removedCount = 0

        if !previous.isEmpty && !workingCurrent.isEmpty && previous.last == workingCurrent.first
            && punctuationTokens.contains(workingCurrent.first!)
        {
            // Remove the duplicate punctuation token from the beginning of current
            workingCurrent = Array(workingCurrent.dropFirst())
            removedCount += 1
        }

        // Check for suffix-prefix overlap: end of previous matches beginning of current
        let maxSearchLength = min(15, previous.count)  // last 15 tokens of previous
        let maxMatchLength = min(maxOverlap, workingCurrent.count)  // first 12 tokens of current

        guard maxSearchLength >= 2 && maxMatchLength >= 2 else {
            return (workingCurrent, removedCount)
        }

        // Search for overlapping sequences from longest to shortest
        for overlapLength in (2...min(maxSearchLength, maxMatchLength)).reversed() {
            // Check if the last `overlapLength` tokens of previous match the first `overlapLength` tokens of current
            let prevSuffix = Array(previous.suffix(overlapLength))
            let currPrefix = Array(workingCurrent.prefix(overlapLength))

            if prevSuffix == currPrefix {
                logger.debug("Found exact suffix-prefix overlap of length \(overlapLength): \(prevSuffix)")
                let finalRemoved = removedCount + overlapLength
                return (Array(workingCurrent.dropFirst(overlapLength)), finalRemoved)
            }
        }

        // Extended search: look for partial overlaps within the sequences
        // Use boundary search frames from TDT config for NeMo-compatible alignment
        let boundarySearchFrames = config.tdtConfig.boundarySearchFrames
        for overlapLength in (2...min(maxSearchLength, maxMatchLength)).reversed() {
            let prevStart = max(0, previous.count - maxSearchLength)
            let prevEnd = previous.count - overlapLength + 1
            if prevEnd <= prevStart { continue }

            for startIndex in prevStart..<prevEnd {
                let prevSub = Array(previous[startIndex..<(startIndex + overlapLength)])
                let currEnd = max(0, workingCurrent.count - overlapLength + 1)

                // Use boundarySearchFrames to limit search window (NeMo tdt_search_boundary pattern)
                let searchLimit = min(boundarySearchFrames, currEnd)
                for currentStart in 0..<searchLimit {
                    let currSub = Array(workingCurrent[currentStart..<(currentStart + overlapLength)])
                    if prevSub == currSub {
                        logger.debug(
                            "Found duplicate sequence length=\(overlapLength) at currStart=\(currentStart): \(prevSub) (boundarySearch=\(boundarySearchFrames))"
                        )
                        let finalRemoved = removedCount + currentStart + overlapLength
                        return (Array(workingCurrent.dropFirst(currentStart + overlapLength)), finalRemoved)
                    }
                }
            }
        }

        return (workingCurrent, removedCount)
    }

    /// Calculate start frame offset for a sliding window segment (deprecated - now handled by timeJump)
    nonisolated internal func calculateStartFrameOffset(segmentIndex: Int, leftContextSeconds: Double) -> Int {
        // This method is deprecated as frame tracking is now handled by the decoder's timeJump mechanism
        // Kept for test compatibility
        return 0
    }

    // MARK: - Vocabulary Rescoring

    /// Apply vocabulary rescoring to an ASRResult using CTC-based constrained decoding.
    ///
    /// Runs CTC inference on the audio samples and applies vocabulary rescoring to correct
    /// misrecognized words. Returns an updated ASRResult with rescored text and populated
    /// `ctcDetectedTerms`/`ctcAppliedTerms` fields.
    ///
    /// - Parameters:
    ///   - result: The original ASRResult from transcription
    ///   - audioSamples: Audio samples used for CTC inference
    /// - Returns: An ASRResult with rescored text and CTC metadata, or the original result if rescoring was skipped
    internal func applyVocabularyRescoring(
        result: ASRResult, audioSamples: [Float]
    ) async -> ASRResult {
        guard let spotter = ctcSpotter,
            let rescorer = vocabularyRescorer,
            let vocab = customVocabulary,
            let tokenTimings = result.tokenTimings, !tokenTimings.isEmpty
        else {
            return result
        }

        do {
            let spotResult = try await spotter.spotKeywordsWithLogProbs(
                audioSamples: audioSamples,
                customVocabulary: vocab,
                minScore: nil
            )

            let logProbs = spotResult.logProbs
            guard !logProbs.isEmpty else {
                logger.debug("Vocabulary rescoring skipped: no log probs from CTC")
                return result
            }

            let vocabConfig = vocabSizeConfig ?? ContextBiasingConstants.rescorerConfig(forVocabSize: 0)
            // Use the higher of the size-based default and the caller-specified threshold
            // so that CustomVocabularyContext.minSimilarity is respected when stricter.
            let effectiveMinSimilarity = max(vocabConfig.minSimilarity, vocab.minSimilarity)

            let rescoreOutput = rescorer.ctcTokenRescore(
                transcript: result.text,
                tokenTimings: tokenTimings,
                logProbs: logProbs,
                frameDuration: spotResult.frameDuration,
                cbw: vocabConfig.cbw,
                marginSeconds: 0.5,
                minSimilarity: effectiveMinSimilarity
            )

            guard rescoreOutput.wasModified else {
                return result
            }

            let detected = rescoreOutput.replacements.compactMap { $0.replacementWord }
            let applied = rescoreOutput.replacements.filter { $0.shouldReplace }.compactMap {
                $0.replacementWord
            }

            logger.info(
                "Vocabulary rescoring applied \(applied.count) replacement(s)"
            )

            return result.withRescoring(
                text: rescoreOutput.text,
                detected: detected.isEmpty ? nil : detected,
                applied: applied.isEmpty ? nil : applied
            )
        } catch {
            logger.warning("Vocabulary rescoring failed: \(error.localizedDescription)")
            return result
        }
    }

    // MARK: - Zipformer2 Inference

    /// Execute inference for Zipformer2 models: mel extraction → encoder → RNNT decode.
    ///
    /// The Zipformer2 encoder takes mel spectrogram frames `[1, T, 80]` as input
    /// (unlike Parakeet which takes raw audio). This method computes the mel features
    /// in Swift, passes them through the CoreML encoder, then runs greedy RNNT decoding.
    internal func executeZipformerInference(
        _ audioSamples: [Float],
        originalLength: Int?,
        actualAudioFrames: Int?,
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int,
        isLastChunk: Bool,
        globalFrameOffset: Int
    ) async throws -> (hypothesis: TdtHypothesis, encoderSequenceLength: Int) {
        guard let melSpec = melSpectrogram,
            let encoderModel = preprocessorModel,  // For Zipformer2, preprocessor IS the encoder
            let models = asrModels
        else {
            throw ASRError.notInitialized
        }

        // Step 1: Compute mel spectrogram from audio samples
        // Zipformer2 uses kaldi-style fbank: 80 bins, no preemphasis, periodic window
        let melResult = melSpec.computeFlatTransposed(audio: audioSamples)
        let melBins = models.version.melBins

        // The encoder has a fixed input size (mel_frames from conversion, default 1495).
        // Read the expected size from the encoder model's input description.
        let encoderInputDesc = encoderModel.modelDescription.inputDescriptionsByName["x"]
        let expectedMelFrames: Int
        if let constraint = encoderInputDesc?.multiArrayConstraint {
            expectedMelFrames = constraint.shape[1].intValue  // [1, T, 80]
        } else {
            expectedMelFrames = 1495  // Default from conversion script
        }

        let actualMelLength = min(melResult.melLength, expectedMelFrames)

        // Step 2: Build encoder input as MLMultiArray [1, expectedMelFrames, melBins]
        // Pad with zeros if audio is shorter, truncate if longer
        let melArray = try MLMultiArray(
            shape: [1, NSNumber(value: expectedMelFrames), NSNumber(value: melBins)],
            dataType: .float32
        )
        // Zero-initialize (handles padding automatically)
        let totalMelElements = expectedMelFrames * melBins
        let melPtr = melArray.dataPointer.bindMemory(to: Float.self, capacity: totalMelElements)
        memset(melPtr, 0, totalMelElements * MemoryLayout<Float>.size)
        // Copy actual mel data (may be shorter than expectedMelFrames)
        let copyElements = min(melResult.mel.count, actualMelLength * melBins)
        melResult.mel.withUnsafeBufferPointer { srcPtr in
            memcpy(melPtr, srcPtr.baseAddress!, copyElements * MemoryLayout<Float>.size)
        }

        let melLensArray = try MLMultiArray(shape: [1], dataType: .int32)
        melLensArray[0] = NSNumber(value: Int32(expectedMelFrames))

        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: melArray),
            "x_lens": MLFeatureValue(multiArray: melLensArray),
        ])

        // Step 3: Run encoder
        try Task.checkCancellation()
        let encoderOutput = try await encoderModel.compatPrediction(
            from: encoderInput, options: predictionOptions)

        let rawEncoderOutput = try extractFeatureValue(
            from: encoderOutput, key: "encoder_out",
            errorMessage: "Invalid Zipformer2 encoder output")
        let encoderLens = try extractFeatureValue(
            from: encoderOutput, key: "encoder_out_lens",
            errorMessage: "Invalid Zipformer2 encoder output length")

        let encoderSequenceLength = encoderLens[0].intValue

        // Step 4: Run RNNT decode
        let hypothesis = try await tdtDecodeWithTimings(
            encoderOutput: rawEncoderOutput,
            encoderSequenceLength: encoderSequenceLength,
            actualAudioFrames: actualAudioFrames ?? encoderSequenceLength,
            originalAudioSamples: audioSamples,
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrameAdjustment,
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset
        )

        return (hypothesis, encoderSequenceLength)
    }
}
