import SwiftUI
import CoreMotion
import TensorFlowLite
import AVFoundation

private let MODELPATH = Bundle.main.path(
    forResource: "model", ofType: "tflite"
)!

typealias Parameter = Float32

protocol Scaler {
    func transform(_ input: [Parameter]) -> [Parameter]
}

private final class PowerTransformer: Scaler {
    enum TransformerError: Error {
        case configFileNotFound
    }

    struct Config: Codable {
        let lambdas: [Parameter]
    }

    let config: Config

    init(configFileName: String) throws {
        guard
            let path = Bundle.main.path(forResource: configFileName, ofType: "json")
        else { throw TransformerError.configFileNotFound }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let config = try JSONDecoder().decode(Config.self, from: data)
        self.config = config
    }

    /// Compute the Yeo-Johnson Power Transformation, as in Scikit-Learn. See: https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-transformer
    func transform(_ input: [Parameter]) -> [Parameter] {
        input.enumerated().map { (i, x) in
            let λ = config.lambdas[i]

            if λ != 0, x >= 0 {
                return (pow(x + 1, λ) - 1) / λ
            } else if λ == 0, x >= 0 {
                return log(x + 1)
            } else if λ != 2, x < 0 {
                return -((pow((-x + 1), 2 - λ) - 1)) / (2 - λ)
            } else /* λ == 2, x < 0 */ {
                return -log(-x + 1)
            }
        }
    }
}

protocol Triaxial {
    var xParam: Parameter { get }
    var yParam: Parameter { get }
    var zParam: Parameter { get }
}

struct ProcessedTriaxial: Triaxial {
    let xParam: Parameter
    let yParam: Parameter
    let zParam: Parameter
}

extension Triaxial {
    func magnitude() -> Parameter {
        sqrt(pow(xParam, 2) + pow(yParam, 2) + pow(zParam, 2))
    }

    func scale(_ f: Parameter) -> ProcessedTriaxial {
        .init(xParam: xParam * f, yParam: yParam * f, zParam: zParam * f)
    }

    func minus(_ other: Triaxial) -> ProcessedTriaxial {
        .init(
            xParam: xParam - other.xParam,
            yParam: yParam - other.yParam,
            zParam: zParam - other.zParam
        )
    }

    func plus(_ other: Triaxial) -> ProcessedTriaxial {
        .init(
            xParam: xParam + other.xParam,
            yParam: yParam + other.yParam,
            zParam: zParam + other.zParam
        )
    }
}

extension CMRotationRate: Triaxial {
    var xParam: Parameter { .init(x) }
    var yParam: Parameter { .init(y) }
    var zParam: Parameter { .init(z) }
}

extension CMAcceleration: Triaxial {
    var xParam: Parameter { .init(x) }
    var yParam: Parameter { .init(y) }
    var zParam: Parameter { .init(z) }
}

extension CMMagneticField: Triaxial {
    var xParam: Parameter { .init(x) }
    var yParam: Parameter { .init(y) }
    var zParam: Parameter { .init(z) }
}

private class Window {
    private let length: Int
    fileprivate var values: [Parameter]

    var isFull: Bool { values.count == length }

    init(length: Int) {
        self.length = length
        self.values = []
    }

    func push(_ value: Parameter) {
        values.append(value)
        if values.count > length {
            values.removeFirst(values.count - length)
        }
    }

    func transform<S>(with scaler: S) -> [Parameter]? where S: Scaler {
        guard isFull else { return nil }
        return scaler.transform(values)
    }
}

private class MotionSource: ObservableObject {
    private let manager: CMMotionManager

    private let accMagScaler: PowerTransformer
    private let magMagScaler: PowerTransformer
    private let gyrMagScaler: PowerTransformer

    private var sampleTimer: Timer?
    private var predictionTimer: Timer?

    init() throws {
        manager = CMMotionManager()

        manager.accelerometerUpdateInterval = 0.01
        manager.deviceMotionUpdateInterval = 0.01
        manager.gyroUpdateInterval = 0.01

        accMagScaler = try .init(configFileName: "acc_mag.scaler")
        magMagScaler = try .init(configFileName: "mag_mag.scaler")
        gyrMagScaler = try .init(configFileName: "gyr_mag.scaler")
    }

    func startListening(
        sampleCallback: @escaping ([(String, Parameter)]) -> Void,
        inferenceCallback: @escaping ([[[Parameter]]]) -> Void
    ) {
        manager.startAccelerometerUpdates()
        manager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical)
        manager.startGyroUpdates()

        let accMagWindow = Window(length: 500)
        let magMagWindow = Window(length: 500)
        let gyrMagWindow = Window(length: 500)

        sampleTimer = Timer(fire: Date(), interval: 0.01, repeats: true) { _ in
            guard
                let accData = self.manager.accelerometerData,
                let motionData = self.manager.deviceMotion,
                let gyrData = self.manager.gyroData
            else { return }

            /// For the original SHL Code, see: https://github.com/STRCWearlab/DataLogger/blob/master/app/src/main/java/uk/ac/sussex/wear/android/datalogger/collector/DataCollectors.java
            /// For information about the Android Sensors, see: https://developer.android.com/reference/android/hardware/SensorEvent#values
            /// For information about the iOS Sensors, see: https://developer.apple.com/documentation/coremotion/cmmotionmanager

            /// `TYPE_ACCELEROMETER` from Android (with included gravity component)
            let accMag = accData.acceleration.scale(9.81).magnitude()
            /// `TYPE_MAGNETIC_FIELD` from Android (calibrated magnetic field)
            let magMag = motionData.magneticField.field.magnitude()
            /// `TYPE_GYROSCOPE` from Android (in rad/s)
            let gyrMag = gyrData.rotationRate.magnitude()

            accMagWindow.push(accMag)
            magMagWindow.push(magMag)
            gyrMagWindow.push(gyrMag)

            sampleCallback([
                ("acc_mag", accMag),
                ("mag_mag", magMag),
                ("gyr_mag", gyrMag),
            ])
        }

        RunLoop.current.add(sampleTimer!, forMode: .default)

        predictionTimer = Timer(fire: Date(), interval: 1, repeats: true) { _ in
            guard
                let scaledAccWindow = accMagWindow.transform(with: self.accMagScaler),
                let scaledMagWindow = magMagWindow.transform(with: self.magMagScaler),
                let scaledGyrWindow = gyrMagWindow.transform(with: self.gyrMagScaler)
            else { return }

            let motion = Array(0..<500).map { i in
                [ scaledAccWindow[i], scaledMagWindow[i], scaledGyrWindow[i] ]
            }

            inferenceCallback([motion])
        }

        RunLoop.current.add(predictionTimer!, forMode: .default)
    }

    func stopListening() {
        sampleTimer?.invalidate()
        predictionTimer?.invalidate()

        manager.stopDeviceMotionUpdates()
        manager.stopAccelerometerUpdates()
        manager.stopMagnetometerUpdates()
    }
}

struct Classification: Identifiable {
    let confidence: Parameter
    let label: String

    var id: String { label }
}

private class Model: ObservableObject {
    private let interpreter: Interpreter
    private let input: Tensor
    private var output: Tensor

    init() throws {
        interpreter = try Interpreter(modelPath: MODELPATH)
        try interpreter.allocateTensors()
        input = try interpreter.input(at: 0)
        output = try interpreter.output(at: 0)
    }

    /// Classify input with shape `(1, 500, 3)`.
    func classify(inputs: [[[Parameter]]]) -> [Classification]? {
        let flattenedInputs = inputs.flatMap { e in e }.flatMap { e in e }
        let inputData = flattenedInputs
            .withUnsafeBufferPointer { ptr in Data(buffer: ptr) }
        do {
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()
            self.output = try interpreter.output(at: 0)
        } catch {
            return nil
        }
        let results = output.data.withUnsafeBytes {
            Array(UnsafeBufferPointer<Parameter>(
                start: $0,
                count: output.data.count/MemoryLayout<Parameter>.stride
            ))
        }

        let labels = ["Null", "Still", "Walking", "Run", "Bike", "Car", "Bus", "Train", "Subway"]

        let zippedResults = zip(labels.indices, results)
        let sortedResults = zippedResults.sorted { $0.1 > $1.1 }
            .prefix(results.count)
        let topInferences = sortedResults
            .map { result in Classification(confidence: result.1, label: labels[result.0]) }

        return topInferences
    }
}


@main
struct SHLModelEvaluationApp: App {
    @StateObject private var model = try! Model()
    @StateObject private var source = try! MotionSource()

    @State var classifications = [Classification]()
    @State var sampleInfo = [(String, Parameter)]()

    var content: some View {
        VStack {
            ForEach(sampleInfo, id: \.0) { s in
                Text("\(s.0): \(s.1)")
            }
            Divider()
            ForEach(classifications) { c in
                Text("\(c.label): \(c.confidence)")
            }
        }
        .padding()
    }

    var body: some Scene {
        WindowGroup {

            content.onAppear(perform: {
                try! AVAudioSession.sharedInstance().setCategory(AVAudioSession.Category.playAndRecord, mode: .default, options: .defaultToSpeaker)
                try! AVAudioSession.sharedInstance().setActive(true, options: .notifyOthersOnDeactivation)

                let synthesizer = AVSpeechSynthesizer()

                source.startListening { sampleInfo in
                    self.sampleInfo = sampleInfo
                } inferenceCallback: { motion in
                    guard
                        let classifications = model.classify(inputs: motion)
                    else { return }

                    if let newBestLabel = classifications.first?.label,
                       self.classifications.first?.label != newBestLabel {
                        let utterance = AVSpeechUtterance(string: newBestLabel)
                        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
                        synthesizer.speak(utterance)
                        print("New best label: \(newBestLabel)")
                    }

                    self.classifications = classifications
                }
            })
            .onDisappear(perform: {
                source.stopListening()
            })
        }
    }
}
