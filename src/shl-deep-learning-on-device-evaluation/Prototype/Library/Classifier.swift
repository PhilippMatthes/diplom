import Foundation
import TensorFlowLite

final class Classifier {
    /// An input of shape `(n_timesteps, n_features)`.
    typealias Input = [[Parameter]]

    private let interpreter: Interpreter
    private let input: Tensor
    private var output: Tensor

    enum Class: Int, CustomStringConvertible, Codable {
        case null = 0
        case still = 1
        case walking = 2
        case run = 3
        case bike = 4
        case car = 5
        case bus = 6
        case train = 7
        case subway = 8

        var description: String {
            switch self {
            case .null: return "Null"
            case .still: return "Still"
            case .walking: return "Walking"
            case .run: return "Run"
            case .bike: return "Bike"
            case .car: return "Car"
            case .bus: return "Bus"
            case .train: return "Train"
            case .subway: return "Subway"
            }
        }

        static let order: [Self] = [
            .null, .still, .walking, .run, .bike, .car, .bus, .train, .subway
        ]
    }

    struct Prediction {
        let confidence: Parameter
        let label: Class
    }

    enum ClassifierError: Error {
        case modelNotFound
    }

    init(modelFileName: String) throws {
        guard let modelPath = Bundle.main.path(
            forResource: modelFileName, ofType: "tflite"
        ) else { throw ClassifierError.modelNotFound }
        interpreter = try Interpreter(modelPath: modelPath)
        try interpreter.allocateTensors()
        input = try interpreter.input(at: 0)
        output = try interpreter.output(at: 0)
    }

    func classify(input: Input) throws -> [Prediction] {
        let inputData = input
            .flatMap { arr in arr }
            .withUnsafeBufferPointer { ptr in Data(buffer: ptr) }

        try interpreter.copy(inputData, toInputAt: 0)
        try interpreter.invoke()
        output = try interpreter.output(at: 0)

        let encodedResults = output.data.withUnsafeBytes {
            Array(UnsafeBufferPointer<Parameter>(
                start: $0, count: output.data.count/MemoryLayout<Parameter>.stride
            ))
        }

        return encodedResults.enumerated()
            .map { i, p in .init(confidence: p, label: Class.order[i]) }
            .sorted { r1, r2 in r1.confidence > r2.confidence }
    }
}
