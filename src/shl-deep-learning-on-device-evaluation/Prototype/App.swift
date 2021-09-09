import SwiftUI
import CoreMotion
import TensorFlowLite
import AVFoundation

@main
struct SHLModelEvaluationApp: App {
    @StateObject private var pipeline = try! Pipeline()
    @State private var visualizationIsActive = false

    private var visualizationView: some View {
        ScrollView {
            VStack {
                if let predictions = pipeline.predictions {
                    LabelsView(predictions: predictions)
                        .frame(height: 512)
                }
                ForEach(Sensor.order, id: \.rawValue) { sensor in
                    if let values = pipeline.sensorWindows[sensor]?.values {
                        WindowView(sensor: sensor, window: values)
                            .frame(height: 128)
                    } else {
                        EmptyView()
                    }
                }
            }
            .padding()
        }
    }

    private var profilingView: some View {
        Color.gray.overlay(
            VStack {
                if let predictions = pipeline.predictions {
                    ForEach(predictions, id: \.label.rawValue) { p -> Text in
                        Text("\(p.label.description): \(p.confidence * 100)")
                    }
                }
            }
            .padding()
        )
        .onAppear(perform: {
            pipeline.run()

            UIScreen.main.brightness = CGFloat(1.0)
        })
        .onDisappear(perform: pipeline.stop)
    }

    var content: some View {
        VStack {
            Button("Switch Visualization", action: { visualizationIsActive.toggle() })
            if visualizationIsActive {
                visualizationView
            } else {
                profilingView
            }
        }
    }

    var body: some Scene {
        WindowGroup {
            content
                .onAppear(perform: {
                    pipeline.run()

                    UIScreen.main.brightness = CGFloat(1.0)
                })
                .onDisappear(perform: pipeline.stop)
        }
    }
}
