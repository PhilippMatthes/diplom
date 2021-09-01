import Foundation

protocol Scaler {
    func transform(_ input: [Parameter]) -> [Parameter]
}

final class PowerTransformer: Scaler {
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
