import Foundation
import CoreMotion

final class SensorSource: ObservableObject {
    private var sampleTimer: Timer?
    private let sampleinterval: TimeInterval = 0.01

    private lazy var manager: CMMotionManager = {
        manager = CMMotionManager()
        manager.accelerometerUpdateInterval = sampleinterval
        manager.deviceMotionUpdateInterval = sampleinterval
        manager.gyroUpdateInterval = sampleinterval
        return manager
    }()

    struct Sample {
        let accMag: Parameter
        let magMag: Parameter
        let gyrMag: Parameter
    }

    /// Load data synchronously from the CMMotionManager.
    /// This includes a mapping of the data gathering from the SHL Android app.
    /// For the original SHL Code, see: https://github.com/STRCWearlab/DataLogger
    /// For information about the Android Sensors, see: https://developer.android.com/reference/android/hardware/SensorEvent#values
    /// For information about the iOS Sensors, see: https://developer.apple.com/documentation/coremotion/cmmotionmanager
    func startListening(onSamplesWith callback: @escaping (Sample) -> Void) {
        manager.startAccelerometerUpdates()
        manager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical)
        manager.startGyroUpdates()

        sampleTimer = Timer(
            fire: Date(), interval: sampleinterval, repeats: true
        ) { _ in
            guard
                let accData = self.manager.accelerometerData,
                let motionData = self.manager.deviceMotion,
                let gyrData = self.manager.gyroData
            else { return }

            /// `TYPE_ACCELEROMETER` from Android (with included gravity component)
            let accMag = accData.acceleration
                .toAndroidFormat()
                .magnitude()
            /// `TYPE_MAGNETIC_FIELD` from Android (calibrated magnetic field)
            let magMag = motionData.magneticField.field
                .toAndroidFormat()
                .magnitude()
            /// `TYPE_GYROSCOPE` from Android (in rad/s)
            let gyrMag = gyrData.rotationRate
                .toAndroidFormat()
                .magnitude()

            callback(.init(accMag: accMag, magMag: magMag, gyrMag: gyrMag))
        }

        RunLoop.current.add(sampleTimer!, forMode: .default)
    }

    func stopListening() {
        sampleTimer?.invalidate()

        manager.stopDeviceMotionUpdates()
        manager.stopAccelerometerUpdates()
        manager.stopMagnetometerUpdates()
    }
}

