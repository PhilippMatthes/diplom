import Foundation

final class Pipeline: ObservableObject {
    private let source = SensorSource()

    private let accMagWindow = try! TransformableWindow("acc_mag.scaler")
    private let magMagWindow = try! TransformableWindow("mag_mag.scaler")
    private let gyrMagWindow = try! TransformableWindow("gyr_mag.scaler")

    private let classifier = try! Classifier(modelFileName: "model")

    private var inferenceTimer: Timer?
    private let inferenceInterval: TimeInterval = 0.5

    @Published var currentSample: SensorSource.Sample?
    @Published var predictions: [Classifier.Prediction]?

    private func update(_ sample: SensorSource.Sample) {
        accMagWindow.push(sample.accMag)
        magMagWindow.push(sample.magMag)
        gyrMagWindow.push(sample.gyrMag)

        // Publish changes to listeners
        currentSample = sample
    }

    private func infer(qos: DispatchQoS.QoSClass = .background) {
        guard
            accMagWindow.isFull, magMagWindow.isFull, gyrMagWindow.isFull
        else { return }

        DispatchQueue.global(qos: qos).async { [weak self] in
            guard let self = self else { return }
            let predictions = try? self.classifier.classify(input: [
                self.accMagWindow.transform().values,
                self.magMagWindow.transform().values,
                self.gyrMagWindow.transform().values
            ])
            DispatchQueue.main.async { self.predictions = predictions }
        }
    }

    func run() {
        source.startListening(onSamplesWith: update)

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
