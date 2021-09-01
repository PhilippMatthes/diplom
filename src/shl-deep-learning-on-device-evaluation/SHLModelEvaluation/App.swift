import SwiftUI
import CoreMotion
import TensorFlowLite
import AVFoundation


@main
struct SHLModelEvaluationApp: App {
    @StateObject private var pipeline = Pipeline()

    private func fmt(_ parameter: Parameter) -> String {
        String(format: "%.2f", parameter)
    }

    var content: some View {
        VStack {
            if let sample = pipeline.currentSample {
                Text("Acc. Mag.: \(fmt(sample.accMag))")
                Text("Mag. Mag.: \(fmt(sample.magMag))")
                Text("Gyr. Mag.: \(fmt(sample.gyrMag))")
            }
            Divider()
            if let predictions = pipeline.predictions {
                ForEach(predictions, id: \.label.rawValue) { p -> Text in
                    Text("\(p.label.rawValue): \(fmt(p.confidence * 100))")
                }
            }
        }
        .padding()
    }

    var body: some Scene {
        WindowGroup {
            Color.gray.overlay(content)
                .onAppear(perform: {
                    pipeline.run()

                    UIScreen.main.brightness = CGFloat(1.0)
                })
                .onDisappear(perform: pipeline.stop)
        }
    }
}
