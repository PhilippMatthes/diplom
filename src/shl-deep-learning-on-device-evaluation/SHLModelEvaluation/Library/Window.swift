import Foundation

class Window {
    fileprivate let maxLength: Int
    fileprivate(set) var values: [Parameter]

    var isFull: Bool { values.count == maxLength }

    init(values: [Parameter], maxLength: Int) {
        self.maxLength = maxLength
        self.values = values
    }

    func push(_ value: Parameter) {
        values.append(value)
        if values.count > maxLength {
            values.removeFirst(values.count - maxLength)
        }
    }
}

final class TransformableWindow: Window {
    private let scaler: PowerTransformer

    init(
        values: [Parameter] = [],
        maxLength: Int = 500,
        _ configFileName: String
    ) throws {
        scaler = try .init(configFileName: configFileName)
        super.init(values: values, maxLength: maxLength)
    }

    func transform() -> Window {
        .init(values: scaler.transform(values), maxLength: maxLength)
    }
}
