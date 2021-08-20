import SwiftUI
import CoreMotion
import TensorFlowLite

private let MODELPATH = Bundle.main.path(forResource: "shl-model.resnet", ofType: "tflite")!

typealias Parameter = Float32

private class StandardScaler {
    enum ScalerError: Error {
        case configFileNotFound
    }

    struct Config: Codable {
        let means: [Parameter]
        let scales: [Parameter]
    }

    let config: Config

    init(configFileName: String) throws {
        guard
            let path = Bundle.main.path(forResource: configFileName, ofType: "json")
        else { throw ScalerError.configFileNotFound }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let config = try JSONDecoder().decode(Config.self, from: data)
        self.config = config
    }

    func transform(_ input: [Parameter]) -> [Parameter] {
        input.enumerated().map { (i, inputPoint) in
            let mean = config.means[i]
            let scale = config.scales[i]
            return (inputPoint - mean) / scale
        }
    }
}

private class MotionSource: ObservableObject {
    private let manager: CMMotionManager
    private let accScaler: StandardScaler
    private let magScaler: StandardScaler
    private let gyrScaler: StandardScaler

    private var sampleTimer: Timer?
    private var predictionTimer: Timer?

    init() throws {
        manager = CMMotionManager()
        manager.accelerometerUpdateInterval = 0.01
        manager.gyroUpdateInterval = 0.01
        manager.magnetometerUpdateInterval = 0.01
        accScaler = try .init(configFileName: "acc-scaler")
        magScaler = try .init(configFileName: "mag-scaler")
        gyrScaler = try .init(configFileName: "gyr-scaler")
    }

    private func mag(x: Parameter, y: Parameter, z: Parameter) -> Parameter {
        return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2))
    }

    func startListening(callback: @escaping ([[[Parameter]]]) -> Void) {
        manager.startAccelerometerUpdates()
        manager.startMagnetometerUpdates()
        manager.startGyroUpdates()

        var accWindow = [Parameter]()
        var magWindow = [Parameter]()
        var gyrWindow = [Parameter]()

        sampleTimer = Timer(fire: Date(), interval: 0.01, repeats: true) { _ in
            guard
                let accData = self.manager.accelerometerData,
                let magData = self.manager.magnetometerData,
                let gyrData = self.manager.gyroData
            else { return }

            accWindow.append(self.mag(
                x: Parameter(accData.acceleration.x),
                y: Parameter(accData.acceleration.y),
                z: Parameter(accData.acceleration.z)
            ))
            if accWindow.count > 500 {
                accWindow.removeFirst(accWindow.count - 500)
            }

            magWindow.append(self.mag(
                x: Parameter(magData.magneticField.x),
                y: Parameter(magData.magneticField.y),
                z: Parameter(magData.magneticField.z)
            ))
            if magWindow.count > 500 {
                magWindow.removeFirst(magWindow.count - 500)
            }

            gyrWindow.append(self.mag(
                x: Parameter(gyrData.rotationRate.x),
                y: Parameter(gyrData.rotationRate.y),
                z: Parameter(gyrData.rotationRate.z)
            ))
            if gyrWindow.count > 500 {
                gyrWindow.removeFirst(gyrWindow.count - 500)
            }
        }

        RunLoop.current.add(self.sampleTimer!, forMode: .default)

        self.predictionTimer = Timer(fire: Date(), interval: 1, repeats: true) { _ in
            let scaledAccWindow = self.accScaler.transform(accWindow)
            let scaledMagWindow = self.magScaler.transform(magWindow)
            let scaledGyrWindow = self.gyrScaler.transform(gyrWindow)

            if
                scaledAccWindow.count == 500,
                scaledMagWindow.count == 500,
                scaledGyrWindow.count == 500
            {
                let motion = Array(0..<500).map { i in
                    [ scaledAccWindow[i], scaledMagWindow[i], scaledGyrWindow[i] ]
                }

                callback([motion])
            }
        }

        RunLoop.current.add(self.predictionTimer!, forMode: .default)
    }

    func stopListening() {
        manager.stopGyroUpdates()
        manager.stopAccelerometerUpdates()
        manager.stopMagnetometerUpdates()

        sampleTimer?.invalidate()
        predictionTimer?.invalidate()
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

    var content: some View {
        VStack {
            HStack(alignment: .bottom) {
                ForEach(classifications) { c in
                    GeometryReader { proxy in
                        VStack {
                            Spacer()
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.blue)
                                .frame(height: proxy.size.height * CGFloat(c.confidence))
                        }
                    }
                }
            }
            HStack(alignment: .bottom) {
                ForEach(classifications) { c in
                    VStack {
                        Text(c.label)
                            .lineLimit(nil)
                        Text("\(Int(c.confidence * 100))%")
                    }
                }
            }
        }
        .padding()
    }

    var body: some Scene {
        WindowGroup {
            content.onAppear(perform: {
                source.startListening { motion in
                    guard
                        let classifications = model.classify(inputs: motion)
                    else { return }
                    withAnimation {
                        self.classifications = classifications
                    }
                }
            })
            .onDisappear(perform: {
                source.stopListening()
            })
        }
    }
}
