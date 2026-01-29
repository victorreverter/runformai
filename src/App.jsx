import { useState, useRef, useEffect } from 'react'
import Webcam from 'react-webcam'
import * as poseDetection from '@tensorflow-models/pose-detection'
import * as tf from '@tensorflow/tfjs-core'
import '@tensorflow/tfjs-backend-webgl'

function App() {
  // Mode state
  const [mode, setMode] = useState('upload') // 'camera' or 'upload'

  // Model state
  const [modelLoading, setModelLoading] = useState(true)

  // Upload mode state
  const [uploadedFile, setUploadedFile] = useState(null)
  const [mediaType, setMediaType] = useState(null) // 'video' or 'image'
  const [mediaDimensions, setMediaDimensions] = useState({ width: 640, height: 480 })
  const [isDragging, setIsDragging] = useState(false)

  // Video control state
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0)

  // Biomechanics state
  const [torsoLean, setTorsoLean] = useState(0)
  const [leanFeedback, setLeanFeedback] = useState('upright')
  const [kneeAngle, setKneeAngle] = useState(0)
  const [hipAngle, setHipAngle] = useState(0)
  const [verticalOscillation, setVerticalOscillation] = useState(0)
  const [headAlignment, setHeadAlignment] = useState(0)
  const [cadence, setCadence] = useState(0)

  // Camera mode state (preserved from original)
  const [webcamError, setWebcamError] = useState(null)
  const [webcamReady, setWebcamReady] = useState(false)

  // Refs
  const webcamRef = useRef(null)
  const videoRef = useRef(null)
  const imageRef = useRef(null)
  const canvasRef = useRef(null)
  const detectorRef = useRef(null)
  const animationIdRef = useRef(null)
  const fileInputRef = useRef(null)

  // Tracking refs for metrics
  const footStrikesRef = useRef([])
  const previousHipHeightRef = useRef(null)

  // Load MoveNet Multipose model
  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.setBackend('webgl')
        await tf.ready()
        console.log('✅ TensorFlow.js backend initialized:', tf.getBackend())

        const detectorConfig = {
          modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
        }
        const detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          detectorConfig
        )
        detectorRef.current = detector
        setModelLoading(false)
        console.log('✅ MoveNet Multipose model loaded successfully')
      } catch (error) {
        console.error('❌ Error loading model:', error)
      }
    }

    loadModel()

    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current)
      }
    }
  }, [])

  // Calculate torso lean angle (FIXED)
  const calculateTorsoLean = (leftShoulder, rightShoulder, leftHip, rightHip) => {
    if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) return 0

    // Calculate midpoints
    const shoulderMid = {
      x: (leftShoulder.x + rightShoulder.x) / 2,
      y: (leftShoulder.y + rightShoulder.y) / 2
    }
    const hipMid = {
      x: (leftHip.x + rightHip.x) / 2,
      y: (leftHip.y + rightHip.y) / 2
    }

    // Calculate angle from vertical (CORRECTED)
    // Positive = forward lean, Negative = backward lean
    const dx = shoulderMid.x - hipMid.x
    const dy = hipMid.y - shoulderMid.y // Inverted for correct direction
    const angleRad = Math.atan2(dx, dy)
    const angleDeg = (angleRad * 180) / Math.PI

    return Math.round(angleDeg)
  }

  // Calculate angle between three points
  const calculateAngle = (point1, point2, point3) => {
    if (!point1 || !point2 || !point3) return 0

    const v1 = { x: point1.x - point2.x, y: point1.y - point2.y }
    const v2 = { x: point3.x - point2.x, y: point3.y - point2.y }

    const angle = Math.abs(
      Math.atan2(v2.y, v2.x) - Math.atan2(v1.y, v1.x)
    )

    let degrees = (angle * 180) / Math.PI
    if (degrees > 180) degrees = 360 - degrees

    return Math.round(degrees)
  }

  // Calculate vertical oscillation
  const calculateVerticalOscillation = (hipY) => {
    if (!hipY) return 0

    if (previousHipHeightRef.current === null) {
      previousHipHeightRef.current = hipY
      return 0
    }

    const oscillation = Math.abs(hipY - previousHipHeightRef.current)
    previousHipHeightRef.current = hipY

    return Math.round(oscillation)
  }

  // Calculate head alignment
  const calculateHeadAlignment = (nose, neck) => {
    if (!nose || !neck) return 0

    const dx = nose.x - neck.x
    const dy = neck.y - nose.y
    const angleRad = Math.atan2(dx, dy)
    const angleDeg = (angleRad * 180) / Math.PI

    return Math.round(angleDeg)
  }

  // Track cadence (steps per minute)
  const trackCadence = (leftAnkle, rightAnkle) => {
    if (!leftAnkle || !rightAnkle) return

    const currentTime = Date.now()
    const leftY = leftAnkle.y
    const rightY = rightAnkle.y

    // Detect foot strike (when ankle is at lowest point)
    const threshold = 5 // pixels
    if (previousHipHeightRef.current &&
      (Math.abs(leftY - previousHipHeightRef.current) < threshold ||
        Math.abs(rightY - previousHipHeightRef.current) < threshold)) {
      footStrikesRef.current.push(currentTime)

      // Keep only last 10 seconds of data
      footStrikesRef.current = footStrikesRef.current.filter(
        time => currentTime - time < 10000
      )

      // Calculate cadence from recent strikes
      if (footStrikesRef.current.length >= 4) {
        const timeSpan = (currentTime - footStrikesRef.current[0]) / 1000 / 60 // minutes
        const stepsPerMinute = Math.round(footStrikesRef.current.length / timeSpan)
        setCadence(stepsPerMinute)
      }
    }
  }

  // Classify lean angle (RECALIBRATED for elite runners)
  const classifyLean = (angle) => {
    if (angle < -5) return 'backward'      // Leaning back
    if (angle >= -5 && angle < 3) return 'upright'  // Too upright
    if (angle >= 3 && angle <= 12) return 'good'    // Optimal (elite runners: 3-8°)
    return 'excessive'                     // Too much forward lean
  }

  // Get feedback message and color
  const getFeedbackInfo = (classification) => {
    const feedbackMap = {
      backward: { message: 'Leaning Backward', color: 'text-neon-red' },
      upright: { message: 'Upright', color: 'text-neon-yellow' },
      good: { message: 'Good Forward Lean', color: 'text-neon-green' },
      excessive: { message: 'Excessive Lean', color: 'text-neon-orange' }
    }
    return feedbackMap[classification] || feedbackMap.upright
  }

  // Draw keypoints on canvas
  const drawKeypoints = (keypoints, ctx) => {
    keypoints.forEach((keypoint) => {
      if (keypoint.score > 0.3) {
        const { x, y } = keypoint
        ctx.beginPath()
        ctx.arc(x, y, 4, 0, 2 * Math.PI)
        ctx.fillStyle = '#00FFFF'  // Cyan for better video visibility
        ctx.fill()
      }
    })
  }

  // Draw skeleton connections
  const drawSkeleton = (keypoints, ctx) => {
    const adjacentPairs = [
      [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // Arms
      [5, 11], [6, 12], [11, 12], // Torso
      [11, 13], [13, 15], [12, 14], [14, 16], // Legs
    ]

    adjacentPairs.forEach(([i, j]) => {
      const kp1 = keypoints[i]
      const kp2 = keypoints[j]

      if (kp1.score > 0.3 && kp2.score > 0.3) {
        ctx.beginPath()
        ctx.moveTo(kp1.x, kp1.y)
        ctx.lineTo(kp2.x, kp2.y)
        ctx.strokeStyle = '#00FFFF'  // Cyan for better video visibility
        ctx.lineWidth = 2
        ctx.stroke()
      }
    })
  }

  // Process poses and update biomechanics
  const processPoses = (poses, ctx) => {
    if (poses.length > 0) {
      const keypoints = poses[0].keypoints

      // Get all relevant keypoints
      const nose = keypoints[0]
      const leftShoulder = keypoints[5]
      const rightShoulder = keypoints[6]
      const leftHip = keypoints[11]
      const rightHip = keypoints[12]
      const leftKnee = keypoints[13]
      const rightKnee = keypoints[14]
      const leftAnkle = keypoints[15]
      const rightAnkle = keypoints[16]

      // Calculate torso lean
      const lean = calculateTorsoLean(leftShoulder, rightShoulder, leftHip, rightHip)
      setTorsoLean(lean)
      setLeanFeedback(classifyLean(lean))

      // Calculate knee angle (right leg)
      if (rightHip && rightKnee && rightAnkle) {
        const knee = calculateAngle(rightHip, rightKnee, rightAnkle)
        setKneeAngle(knee)
      }

      // Calculate hip angle (right leg)
      if (rightShoulder && rightHip && rightKnee) {
        const hip = calculateAngle(rightShoulder, rightHip, rightKnee)
        setHipAngle(hip)
      }

      // Calculate vertical oscillation
      const hipMidY = (leftHip.y + rightHip.y) / 2
      const oscillation = calculateVerticalOscillation(hipMidY)
      setVerticalOscillation(oscillation)

      // Calculate head alignment
      const neckY = (leftShoulder.y + rightShoulder.y) / 2
      const neck = { x: (leftShoulder.x + rightShoulder.x) / 2, y: neckY }
      const headAlign = calculateHeadAlignment(nose, neck)
      setHeadAlignment(headAlign)

      // Track cadence
      trackCadence(leftAnkle, rightAnkle)

      // Draw pose
      drawKeypoints(keypoints, ctx)
      drawSkeleton(keypoints, ctx)
    }
  }

  // Video frame detection loop
  const detectVideoFrame = async () => {
    if (
      detectorRef.current &&
      videoRef.current &&
      !videoRef.current.paused &&
      !videoRef.current.ended
    ) {
      const video = videoRef.current
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')

      // Detect poses
      const poses = await detectorRef.current.estimatePoses(video)

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Process and draw
      processPoses(poses, ctx)

      // Continue loop
      animationIdRef.current = requestAnimationFrame(detectVideoFrame)
    }
  }

  // Image detection
  const detectImage = async () => {
    if (detectorRef.current && imageRef.current) {
      const image = imageRef.current
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')

      // Detect poses
      const poses = await detectorRef.current.estimatePoses(image)

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Process and draw
      processPoses(poses, ctx)
    }
  }

  // Camera mode detection loop (preserved)
  const runMovenet = async () => {
    if (
      detectorRef.current &&
      webcamRef.current &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')

      const poses = await detectorRef.current.estimatePoses(video)
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      if (poses.length > 0) {
        const keypoints = poses[0].keypoints

        // Calculate torso lean for camera mode too
        const leftShoulder = keypoints[5]
        const rightShoulder = keypoints[6]
        const leftHip = keypoints[11]
        const rightHip = keypoints[12]

        const lean = calculateTorsoLean(leftShoulder, rightShoulder, leftHip, rightHip)
        setTorsoLean(lean)
        setLeanFeedback(classifyLean(lean))

        drawKeypoints(keypoints, ctx)
        drawSkeleton(keypoints, ctx)
      }
    }

    animationIdRef.current = requestAnimationFrame(runMovenet)
  }

  // Start camera detection when model is loaded
  useEffect(() => {
    if (!modelLoading && mode === 'camera' && webcamRef.current) {
      runMovenet()
    }
    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current)
      }
    }
  }, [modelLoading, mode])

  // Handle file upload
  const handleFileUpload = (file) => {
    if (!file) return

    const fileType = file.type
    if (fileType.startsWith('video/')) {
      setMediaType('video')
      setUploadedFile(URL.createObjectURL(file))
    } else if (fileType.startsWith('image/')) {
      setMediaType('image')
      setUploadedFile(URL.createObjectURL(file))
    } else {
      alert('Please upload a video (.mp4) or image (.jpg, .png) file')
    }
  }

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    handleFileUpload(file)
  }

  // Video event handlers
  const handleVideoLoaded = (e) => {
    const video = e.target
    const width = video.videoWidth
    const height = video.videoHeight
    setMediaDimensions({ width, height })

    if (canvasRef.current) {
      canvasRef.current.width = width
      canvasRef.current.height = height
    }
  }

  const handleImageLoaded = (e) => {
    const image = e.target
    const width = image.naturalWidth
    const height = image.naturalHeight
    setMediaDimensions({ width, height })

    if (canvasRef.current) {
      canvasRef.current.width = width
      canvasRef.current.height = height
    }

    // Run detection on image
    setTimeout(() => detectImage(), 100)
  }

  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
        if (animationIdRef.current) {
          cancelAnimationFrame(animationIdRef.current)
        }
      } else {
        videoRef.current.play()
        detectVideoFrame()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const toggleSpeed = () => {
    const newSpeed = playbackSpeed === 1.0 ? 0.5 : 1.0
    setPlaybackSpeed(newSpeed)
    if (videoRef.current) {
      videoRef.current.playbackRate = newSpeed
    }
  }

  if (modelLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <img src="/logo.png" alt="RunForm AI" className="h-16 mb-4 mx-auto" />
          <p className="text-2xl text-neon-green animate-pulse">
            Loading AI Model...
          </p>
        </div>
      </div>
    )
  }

  const feedbackInfo = getFeedbackInfo(leanFeedback)

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center justify-center p-6">
      {/* Header */}
      <div className="mb-4">
        <img src="/logo.png" alt="RunForm AI" className="h-12" />
      </div>

      {/* Mode Toggle */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={() => setMode('camera')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${mode === 'camera'
            ? 'bg-purple-600 text-white'
            : 'bg-slate-800 text-gray-400 hover:bg-slate-700'
            }`}>
          📹 Live Camera
        </button>
        <button
          onClick={() => setMode('upload')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${mode === 'upload'
            ? 'bg-purple-600 text-white'
            : 'bg-slate-800 text-gray-400 hover:bg-slate-700'
            }`}>
          📁 Upload File
        </button>
      </div>

      {/* Upload Mode */}
      {mode === 'upload' && (
        <div className="w-full max-w-4xl">
          {!uploadedFile ? (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-4 border-dashed rounded-lg p-16 text-center cursor-pointer transition-all ${isDragging
                ? 'border-purple-400 bg-purple-900/20'
                : 'border-purple-700 hover:border-purple-500'
                }`}>
              <p className="text-3xl text-purple-400 mb-4">📤</p>
              <p className="text-xl text-purple-400 mb-2">
                Drag & Drop Your Running Video or Image
              </p>
              <p className="text-sm text-gray-500">
                Supports .mp4, .jpg, .png
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4,image/jpeg,image/png"
                onChange={(e) => handleFileUpload(e.target.files[0])}
                className="hidden"
              />
            </div>
          ) : (
            <div className="space-y-4 flex flex-col items-center">
              {/* Media Container */}
              <div className="relative">
                {mediaType === 'video' ? (
                  <video
                    ref={videoRef}
                    src={uploadedFile}
                    onLoadedMetadata={handleVideoLoaded}
                    className="rounded-lg max-w-full"
                    style={{ maxHeight: '60vh' }}
                  />
                ) : (
                  <img
                    ref={imageRef}
                    src={uploadedFile}
                    onLoad={handleImageLoaded}
                    className="rounded-lg max-w-full"
                    style={{ maxHeight: '60vh' }}
                    alt="Uploaded running form"
                  />
                )}
                <canvas
                  ref={canvasRef}
                  className="absolute top-0 left-0 rounded-lg pointer-events-none"
                />
              </div>

              {/* Video Controls */}
              {mediaType === 'video' && (
                <div className="flex gap-4 justify-center">
                  <button
                    onClick={togglePlayPause}
                    className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-bold transition-colors">
                    {isPlaying ? '⏸ Pause' : '▶ Play'}
                  </button>
                  <button
                    onClick={toggleSpeed}
                    className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-bold transition-colors">
                    🐌 {playbackSpeed}x Speed
                  </button>
                </div>
              )}

              {/* Change File Button */}
              <div className="text-center">
                <button
                  onClick={() => {
                    setUploadedFile(null)
                    setMediaType(null)
                    setIsPlaying(false)
                  }}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors">
                  📁 Upload Different File
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Camera Mode */}
      {mode === 'camera' && (
        <div className="relative mb-8">
          <Webcam
            ref={webcamRef}
            width={640}
            height={480}
            videoConstraints={{
              width: 640,
              height: 480,
              facingMode: 'user',
            }}
            onUserMedia={() => setWebcamReady(true)}
            onUserMediaError={(err) => setWebcamError(err.message)}
            className="rounded-lg"
          />

          {!webcamReady && !webcamError && (
            <div className="absolute inset-0 bg-gray-900 rounded-lg flex items-center justify-center">
              <p className="text-xl text-purple-400 animate-pulse">
                Requesting camera access...
              </p>
            </div>
          )}

          {webcamError && (
            <div className="absolute inset-0 bg-gray-900 rounded-lg flex items-center justify-center p-8">
              <div className="text-center">
                <p className="text-xl text-neon-red mb-4">
                  Camera Access Denied
                </p>
                <p className="text-sm text-green-400 mb-4">
                  Please allow camera access in your browser settings
                </p>
                <button
                  onClick={() => window.location.reload()}
                  className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-bold transition-colors">
                  Retry
                </button>
              </div>
            </div>
          )}

          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className="absolute top-0 left-0 rounded-lg"
          />
        </div>
      )}

      {/* Biomechanics HUD */}
      <div className="mt-6 bg-slate-800/40 rounded-lg p-4 border border-purple-700/30 max-w-4xl">
        <h2 className="text-lg text-purple-400 mb-3 text-center font-semibold">
          Biomechanics Analysis
        </h2>

        {/* Metrics Grid - 2 columns */}
        <div className="grid grid-cols-2 gap-2.5">
          {/* Torso Lean */}
          <div className="text-center bg-slate-800/20 p-2.5 rounded-lg">
            <p className="text-xs text-purple-400 mb-1">TORSO LEAN</p>
            <p className={`text-2xl font-bold ${feedbackInfo.color}`}>
              {torsoLean}°
            </p>
            <p className={`text-xs mt-1 ${feedbackInfo.color}`}>
              {feedbackInfo.message}
            </p>
          </div>

          {/* Vertical Oscillation */}
          <div className="text-center bg-slate-800/20 p-2.5 rounded-lg">
            <p className="text-xs text-purple-400 mb-1">VERT. OSC.</p>
            <p className="text-2xl font-bold text-purple-400">
              {verticalOscillation}px
            </p>
            <p className="text-xs mt-1 text-gray-400">
              {verticalOscillation < 40 ? 'Excellent' : verticalOscillation < 60 ? 'Good' : 'High'}
            </p>
          </div>

          {/* Knee Angle */}
          <div className="text-center bg-slate-800/15 p-2.5 rounded-lg">
            <p className="text-xs text-purple-400 mb-1">KNEE</p>
            <p className="text-xl font-bold text-green-400">
              {kneeAngle}°
            </p>
          </div>

          {/* Hip Angle */}
          <div className="text-center bg-slate-800/15 p-2.5 rounded-lg">
            <p className="text-xs text-purple-400 mb-1">HIP</p>
            <p className="text-xl font-bold text-green-400">
              {hipAngle}°
            </p>
          </div>

          {/* Head Alignment */}
          <div className="text-center bg-slate-800/15 p-2.5 rounded-lg">
            <p className="text-xs text-purple-400 mb-1">HEAD</p>
            <p className="text-xl font-bold text-green-400">
              {headAlignment}°
            </p>
          </div>

          {/* Cadence */}
          <div className="text-center bg-slate-800/15 p-2.5 rounded-lg">
            <p className="text-xs text-purple-400 mb-1">CADENCE</p>
            <p className="text-xl font-bold text-green-400">
              {cadence} SPM
            </p>
          </div>
        </div>

        {/* Elite Reference */}
        <div className="mt-3 text-center text-xs text-gray-500">
          Elite: Torso 3-8° · Osc. &lt;40px · Cadence 180+ SPM
        </div>
      </div>

      {/* Instructions */}
      <div className="mt-4 text-center max-w-2xl">
        <p className="text-xs text-gray-400">
          {mode === 'upload'
            ? 'Upload a video or image of your running form to analyze torso lean angle'
            : 'Stand in front of the camera to analyze your running form in real-time'}
        </p>
      </div>
    </div>
  )
}

export default App
