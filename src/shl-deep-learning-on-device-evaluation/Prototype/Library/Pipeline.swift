import Foundation

final class Pipeline: ObservableObject {
    private let source: SensorSource
    private let sensorWindows: [Sensor: Window]
    private let sensorPreprocessors: [Sensor: [Preprocessor]]
    private let classifier: Classifier
    private let inferenceInterval: TimeInterval
    private var inferenceTimer: Timer?

    @Published var predictions: [Classifier.Prediction]?

    init(
        inferenceInterval: TimeInterval = 1,
        sampleInterval: TimeInterval = 0.01,
        windowLength: Int = 500,
        movingAveragePeriod: Int = 5,
        modelFileName: String = "model"
    ) throws {
        self.inferenceInterval = inferenceInterval

        source = SensorSource(sampleInterval: sampleInterval)
        sensorWindows = Dictionary(uniqueKeysWithValues: Sensor.order.map { sensor in
            let window = Window(maxLength: windowLength)
            return (sensor, window)
        })
        sensorPreprocessors = Dictionary(uniqueKeysWithValues: try Sensor.order.map { sensor in
            let preprocessors: [Preprocessor] = [
                try PowerTransformer(sensor: sensor),
                // MovingAverage(period: movingAveragePeriod),
            ]
            return (sensor, preprocessors)
        })
        classifier = try Classifier(modelFileName: modelFileName)
    }

    private func infer(qos: DispatchQoS.QoSClass = .background) {
        DispatchQueue.global(qos: qos).async {
            for window in self.sensorWindows.values {
                guard window.isFull else { return }
            }

            var input = [[Parameter]]()
            for sensor in Sensor.order {
                guard
                    var values = self.sensorWindows[sensor]?.values,
                    let preprocessors = self.sensorPreprocessors[sensor]
                else { return }
                for preprocessor in preprocessors {
                    values = preprocessor.transform(values)
                }
                input.append(values)
            }

            guard
                let predictions = try? self.classifier.classify(input: input)
            else { return }
            DispatchQueue.main.async { self.predictions = predictions }
        }
    }

    func run() {
        source.startListening { samples in
            for (sensor, sample) in samples {
                self.sensorWindows[sensor]?.push(sample)
            }
        }

        inferenceTimer = Timer(
            fire: Date(), interval: inferenceInterval, repeats: true
        ) { _ in self.infer() }

        RunLoop.current.add(inferenceTimer!, forMode: .default)
    }

    func stop() {
        inferenceTimer?.invalidate()

        source.stopListening()
    }
}
