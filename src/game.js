import * as THREE from 'three';
import * as OBC from '@thatopen/components';
import { ImageSegmenter, FilesetResolver } from '@mediapipe/tasks-vision';
import { AudioManager } from './audioManager.js';
import { HandControl, GestureState } from './handControl';

// Must match the installed web-ifc version (node_modules/web-ifc/package.json) —
// see the comment in _setupIfc() for why this can't be auto-detected.
const WEB_IFC_VERSION = '0.0.77';

// Control-value -> scene-transform sensitivities. HandControl emits normalized,
// application-agnostic deltas; these tune how "big" they feel in this viewer.
const ROTATE_SENSITIVITY = 6; // radians per unit of normalized dx/dy
const PAN_SENSITIVITY = 1; // world-pixels per unit of normalized dx/dy * viewport size
const SECTION_SENSITIVITY = 2; // world-pixels per unit of normalized delta * viewport height
const ZOOM_MIN_FACTOR = 0.2;
const ZOOM_MAX_FACTOR = 5;

const SECTION_ORIENTATIONS = [
  { label: 'Horizontal', normal: new THREE.Vector3(0, 1, 0) },
  { label: 'Vertical X', normal: new THREE.Vector3(1, 0, 0) },
  { label: 'Vertical Z', normal: new THREE.Vector3(0, 0, 1) },
];

const GESTURE_STATE_COLORS = {
  [GestureState.IDLE]: '#888888',
  [GestureState.HAND_VISIBLE]: '#aaaaaa',
  [GestureState.POINTING]: '#00BFFF',
  [GestureState.PINCHING]: '#00FFFF',
  [GestureState.ROTATING]: '#00FFFF',
  [GestureState.PANNING]: '#FF00FF',
  [GestureState.TWO_HAND_ZOOM]: '#FFFF00',
  [GestureState.SECTION_ACTIVE]: '#FFA500',
};

const GESTURE_STATE_LABELS = {
  [GestureState.IDLE]: 'No hand',
  [GestureState.HAND_VISIBLE]: 'Hand ready',
  [GestureState.POINTING]: 'Pointing',
  [GestureState.PINCHING]: 'Pinch held',
  [GestureState.ROTATING]: 'Rotating',
  [GestureState.PANNING]: 'Panning',
  [GestureState.TWO_HAND_ZOOM]: 'Two-hand zoom',
  [GestureState.SECTION_ACTIVE]: 'Section cut',
  [GestureState.PAUSE_TOGGLE]: 'Pausing gestures',
  [GestureState.RESET_VIEW]: 'Resetting view',
};

const SEGMENTER_MODEL_PATH =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite';
const SEGMENTER_WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const BACKGROUND_BLUR_PX = 24;

export class Game {
  constructor(renderDiv) {
    this.renderDiv = renderDiv;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.videoElement = null;
    this.backgroundVisible = true;
    this.backgroundBlurred = false;
    this.backgroundCanvas = null;
    this.backgroundSourceCanvas = null;
    this.backgroundMaskCanvas = null;
    this.backgroundSegmenter = null;
    this.backgroundSegmenterLoading = false;
    this.lastSegmentationTime = -1;

    this.gameState = 'loading'; // loading, tracking, error
    this.clock = new THREE.Clock();
    this.audioManager = new AudioManager();

    // IFC loading (That Open Components / Fragments)
    this.obcComponents = null;
    this.fragments = null;
    this.ifcLoader = null;
    this.currentModel = null; // THREE.Group wrapping the current fragments model, gesture target
    this.currentFragmentsModel = null; // FRAGS.FragmentsModel, for disposal
    this.currentFragmentsModelId = null;
    this.baseModelScale = 1;

    this.modelMinZ = -200;
    this.modelMaxZ = 50;
    this.modelDefaultZ = -100;

    // Hand control (standalone gesture engine, see src/handControl)
    this.handControl = null;

    // Section / clipping plane
    this.sectionPlane = new THREE.Plane(SECTION_ORIENTATIONS[0].normal.clone(), 0);
    this.sectionOrientationIndex = 0;
    this.sectionOffset = 0;
    this.sectionEnabled = false;
    this.sectionPlaneVisual = null;

    this._init().catch((error) => {
      console.error('Initialization failed:', error);
      this._showError('Initialization failed. Check console.');
    });
  }

  async _init() {
    this._setupDOM();
    this._setupThree();
    await this._setupIfc();
    await this._setupWebcam();
    await this._setupBackgroundSegmenter();
    await this._setupHandControl();
    await this.videoElement.play();

    this.audioManager.resumeContext();
    this.clock.start();
    window.addEventListener('resize', this._onResize.bind(this));
    window.addEventListener('keydown', this._onKeyDown.bind(this));

    this.gameState = 'tracking';
    this._animate();
  }

  // ---------------------------------------------------------------------
  // DOM / UI setup
  // ---------------------------------------------------------------------

  _setupDOM() {
    this.videoElement = document.createElement('video');
    Object.assign(this.videoElement.style, {
      position: 'absolute',
      top: '0',
      left: '0',
      width: '100%',
      height: '100%',
      objectFit: 'cover',
      transform: 'scaleX(-1)', // mirror for intuitive control
      zIndex: '0',
    });
    this.videoElement.autoplay = true;
    this.videoElement.muted = true;
    this.videoElement.playsInline = true;
    this.renderDiv.appendChild(this.videoElement);

    this.backgroundCanvas = document.createElement('canvas');
    this.backgroundCanvas.className = 'background-canvas';
    this.backgroundCanvas.style.display = 'none';
    this.backgroundSourceCanvas = document.createElement('canvas');
    this.backgroundMaskCanvas = document.createElement('canvas');
    this.renderDiv.appendChild(this.backgroundCanvas);

    this._setupStatusOverlay();
    this._setupGestureBadge();
    this._setupGestureControls();
    this._setupLandmarkOverlay();
    this._setupCursorDot();
    this._setupLegend();
    this._setupDragAndDrop();
  }

  _setupStatusOverlay() {
    this.statusContainer = document.createElement('div');
    Object.assign(this.statusContainer.style, {
      position: 'absolute',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      zIndex: '10',
      pointerEvents: 'none',
      textAlign: 'center',
      color: 'white',
      fontFamily: '"Arial", "Helvetica Neue", Helvetica, sans-serif',
      maxWidth: '80vw',
    });

    this.statusText = document.createElement('div');
    this.statusText.style.fontSize = 'clamp(20px, 4vw, 36px)';
    this.statusText.style.fontWeight = 'bold';
    this.statusText.style.textShadow = '2px 2px 4px black';
    this.statusText.innerText = 'Drag & drop an .ifc file to load a model';
    this.statusContainer.appendChild(this.statusText);

    this.renderDiv.appendChild(this.statusContainer);
  }

  _setupGestureBadge() {
    this.gestureBadge = document.createElement('div');
    Object.assign(this.gestureBadge.style, {
      position: 'absolute',
      top: '10px',
      right: '10px',
      zIndex: '30',
      padding: '10px 18px',
      fontFamily: '"Arial", "Helvetica Neue", Helvetica, sans-serif',
      fontWeight: 'bold',
      fontSize: '16px',
      border: '2px solid black',
      borderRadius: '4px',
      boxShadow: '2px 2px 0px black',
      backgroundColor: '#888888',
      color: '#000000',
      textAlign: 'center',
    });
    this.gestureBadge.innerText = GestureState.IDLE;
    this.renderDiv.appendChild(this.gestureBadge);

    this.sectionBadge = document.createElement('div');
    Object.assign(this.sectionBadge.style, {
      position: 'absolute',
      top: '54px',
      right: '10px',
      zIndex: '30',
      padding: '8px 14px',
      fontFamily: '"Arial", "Helvetica Neue", Helvetica, sans-serif',
      fontWeight: 'bold',
      fontSize: '14px',
      border: '2px solid black',
      borderRadius: '4px',
      boxShadow: '2px 2px 0px black',
      backgroundColor: 'white',
      color: '#000000',
      textAlign: 'center',
    });
    this._updateSectionBadgeText();
    this.renderDiv.appendChild(this.sectionBadge);
  }

  _setupGestureControls() {
    this.gestureControls = document.createElement('div');
    this.gestureControls.className = 'gesture-controls';

    this.pauseButton = document.createElement('button');
    this.pauseButton.type = 'button';
    this.pauseButton.innerText = 'Pause gestures';
    this.pauseButton.title = 'Pause or resume hand tracking (Space)';
    this.pauseButton.addEventListener('click', () => this._toggleGesturePause());

    this.resetButton = document.createElement('button');
    this.resetButton.type = 'button';
    this.resetButton.innerText = 'Reset view';
    this.resetButton.title = 'Reset model position, rotation, zoom, and section (R)';
    this.resetButton.addEventListener('click', () => this._resetView());

    this.backgroundButton = document.createElement('button');
    this.backgroundButton.type = 'button';
    this.backgroundButton.title = 'Cycle camera background: clear, blurred, or hidden (B)';
    this.backgroundButton.addEventListener('click', () => this._toggleBackground());

    this.landmarkButton = document.createElement('button');
    this.landmarkButton.type = 'button';
    this.landmarkButton.title = 'Show or hide finger movement lines (L)';
    this.landmarkButton.addEventListener('click', () => this._toggleLandmarkOverlay());

    this.infoButton = document.createElement('button');
    this.infoButton.type = 'button';
    this.infoButton.innerText = 'ⓘ Info';
    this.infoButton.title = 'Show or hide the gesture legend (I)';
    this.infoButton.setAttribute('aria-pressed', 'false');
    this.infoButton.addEventListener('click', () => this._toggleInfoPanel());

    this.gestureControls.append(
      this.pauseButton,
      this.resetButton,
      this.backgroundButton,
      this.landmarkButton,
      this.infoButton
    );
    this._updateBackgroundButton();
    this._updateLandmarkButton();
    this.renderDiv.appendChild(this.gestureControls);

    document.addEventListener('click', (event) => {
      if (!this.infoPanelOpen) return;
      if (event.target === this.infoButton || this.legendText?.contains(event.target)) return;
      this._closeInfoPanel();
    });
  }

  _setupCursorDot() {
    this.cursorDot = document.createElement('div');
    Object.assign(this.cursorDot.style, {
      position: 'absolute',
      top: '0',
      left: '0',
      width: '18px',
      height: '18px',
      marginLeft: '-9px',
      marginTop: '-9px',
      borderRadius: '50%',
      border: '2px solid black',
      backgroundColor: 'rgba(0, 191, 255, 0.6)',
      zIndex: '20',
      pointerEvents: 'none',
      display: 'none',
      transition: 'transform 0.15s ease-out',
    });
    this.renderDiv.appendChild(this.cursorDot);
  }

  _setupLegend() {
    this.legendText = document.querySelector('#instruction-text');
    if (!this.legendText) return;
    this.legendText.innerHTML =
      'Pinch+drag: rotate &nbsp;·&nbsp; Two hands: zoom &nbsp;·&nbsp; Open palm: pan &nbsp;·&nbsp; ' +
      'Point: cursor &nbsp;·&nbsp; Pinch tap: select &nbsp;·&nbsp; ✌ two fingers: section cut ' +
      '&nbsp;·&nbsp; Thumbs up: pause &nbsp;·&nbsp; Fist: reset &nbsp;·&nbsp; Space/R: pause/reset &nbsp;·&nbsp; Keys 1/2/3: section orientation' +
      '&nbsp;·&nbsp; Cycle camera background: clear, blurred, or hidden (B)';
    this.infoPanelOpen = false;
    this.legendText.style.display = 'none';
  }

  _toggleInfoPanel() {
    if (this.infoPanelOpen) this._closeInfoPanel();
    else this._openInfoPanel();
  }

  _openInfoPanel() {
    if (!this.legendText) return;
    this.infoPanelOpen = true;
    this.legendText.style.display = 'block';
    this.infoButton?.setAttribute('aria-pressed', 'true');
  }

  _closeInfoPanel() {
    if (!this.legendText) return;
    this.infoPanelOpen = false;
    this.legendText.style.display = 'none';
    this.infoButton?.setAttribute('aria-pressed', 'false');
  }

  _setupDragAndDrop() {
    this.renderDiv.addEventListener('dragover', (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = 'copy';
      this.renderDiv.style.border = '2px dashed #007bff';
    });
    this.renderDiv.addEventListener('dragleave', () => {
      this.renderDiv.style.border = 'none';
    });
    this.renderDiv.addEventListener('drop', (event) => {
      event.preventDefault();
      this.renderDiv.style.border = 'none';
      if (!event.dataTransfer.files || event.dataTransfer.files.length === 0) return;

      const file = event.dataTransfer.files[0];
      if (file.name.toLowerCase().endsWith('.ifc')) {
        this._loadIfcFile(file);
      } else {
        this._showStatusScreen(`"${file.name}" is not an IFC model.`, 'orange');
        setTimeout(() => this._updateStatusOverlayVisibility(), 3000);
      }
      event.dataTransfer.clearData();
    });
  }

  _setupThree() {
    const width = this.renderDiv.clientWidth;
    const height = this.renderDiv.clientHeight;

    this.scene = new THREE.Scene();

    // Orthographic camera with 1 world-unit == 1 screen pixel, matching the
    // normalized-to-pixel control math applied to gesture deltas.
    this.camera = new THREE.OrthographicCamera(width / -2, width / 2, height / 2, height / -2, 1, 2000);
    this.camera.position.z = 100;

    this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.localClippingEnabled = true;
    Object.assign(this.renderer.domElement.style, { position: 'absolute', top: '0', left: '0', zIndex: '1' });
    this.renderDiv.appendChild(this.renderer.domElement);

    this.scene.add(new THREE.AmbientLight(0xffffff, 1.5));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.8);
    directionalLight.position.set(0, 0, 100);
    this.scene.add(directionalLight);

    const visualGeometry = new THREE.PlaneGeometry(4000, 4000);
    const visualMaterial = new THREE.MeshBasicMaterial({
      color: 0xffa500,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.sectionPlaneVisual = new THREE.Mesh(visualGeometry, visualMaterial);
    this.sectionPlaneVisual.visible = false;
    this.scene.add(this.sectionPlaneVisual);
    this._rebuildSectionPlane();
  }

  // ---------------------------------------------------------------------
  // IFC loading (That Open Components)
  // ---------------------------------------------------------------------

  async _setupIfc() {
    this.obcComponents = new OBC.Components();

    this.fragments = this.obcComponents.get(OBC.FragmentsManager);
    const workerUrl = await OBC.FragmentsManager.getWorker();
    this.fragments.init(workerUrl);

    this.ifcLoader = this.obcComponents.get(OBC.IfcLoader);
    await this.ifcLoader.setup({ autoSetWasm: false });
    this.ifcLoader.settings.wasm.path = `https://unpkg.com/web-ifc@${WEB_IFC_VERSION}/`;
    this.ifcLoader.settings.wasm.absolute = true;

    this.obcComponents.init();
  }

  async _loadIfcFile(file) {
    this._showStatusScreen(`Loading "${file.name}"...`, 'white');
    try {
      const buffer = await file.arrayBuffer();
      await this._parseAndLoadIfc(buffer, file.name);
      this._updateStatusOverlayVisibility();
    } catch (error) {
      console.error(`Error loading IFC model ${file.name}:`, error);
      this._showStatusScreen(`Failed to load "${file.name}". Check console for details.`, 'orange');
    }
  }

  async _parseAndLoadIfc(arrayBuffer, fileName) {
    this._disposeCurrentModel();

    const modelId = `${fileName}-${Date.now()}`;
    const data = new Uint8Array(arrayBuffer);
    const fragmentsModel = await this.ifcLoader.load(data, true, modelId);

    // Fragments streams tile geometry only for what's inside the registered
    // camera's frustum. Real-world IFC files often place geometry far from
    // (0,0,0) (survey/site coordinates), which our small display frustum
    // wouldn't see — so discover the true bounding box with an effectively
    // unbounded camera first, before anything has been culled out.
    const discoveryCamera = new THREE.OrthographicCamera(-1e8, 1e8, 1e8, -1e8, -1e8, 1e8);
    fragmentsModel.useCamera(discoveryCamera);
    await this.fragments.core.update(true);

    this.currentFragmentsModel = fragmentsModel;
    this.currentFragmentsModelId = modelId;
    this.currentModel = this._fitModelToView(fragmentsModel);
    this._applySectionClipping(fragmentsModel.object);
    this.scene.add(this.currentModel);

    // Now that the model's world position sits inside the real display
    // camera's frustum, re-stream tiles against that camera for rendering.
    fragmentsModel.useCamera(this.camera);
    await this.fragments.core.update(true);
  }

  // Centers and uniformly scales the loaded model so it fits comfortably in
  // the ortho camera's screen-pixel frustum, regardless of the model's
  // original real-world units.
  _fitModelToView(fragmentsModel) {
    const object = fragmentsModel.object;
    const box = fragmentsModel.box?.isEmpty?.() === false
      ? fragmentsModel.box
      : new THREE.Box3().setFromObject(object);

    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    object.position.sub(center); // re-center object's own content at local origin

    const maxDimension = Math.max(size.x, size.y, size.z, 0.0001);
    const targetPixelSize = Math.min(this.renderDiv.clientWidth, this.renderDiv.clientHeight) * 0.6;
    const scale = targetPixelSize / maxDimension;
    this.baseModelScale = scale;

    const wrapper = new THREE.Group();
    wrapper.add(object);
    wrapper.scale.set(scale, scale, scale);
    wrapper.position.set(0, 0, this.modelDefaultZ);
    return wrapper;
  }

  _applySectionClipping(object) {
    object.traverse((child) => {
      if (!child.isMesh || !child.material) return;
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((material) => {
        material.clippingPlanes = this.sectionEnabled ? [this.sectionPlane] : [];
        material.clipShadows = true;
        material.needsUpdate = true;
      });
    });
  }

  _disposeCurrentModel() {
    if (this.currentModel) {
      this.scene.remove(this.currentModel);
      this.currentModel = null;
    }
    if (this.currentFragmentsModelId) {
      this.fragments.core.disposeModel(this.currentFragmentsModelId).catch((e) => {
        console.warn('Error disposing previous IFC model:', e);
      });
      this.currentFragmentsModelId = null;
    }
    this.currentFragmentsModel = null;
  }

  _updateStatusOverlayVisibility() {
    if (this.currentModel) {
      this.statusContainer.style.display = 'none';
    } else {
      this.statusContainer.style.display = 'block';
      this.statusText.innerText = 'Drag & drop an .ifc file to load a model';
      this.statusText.style.color = 'white';
    }
  }

  _showStatusScreen(message, color = 'white') {
    this.statusContainer.style.display = 'block';
    this.statusText.innerText = message;
    this.statusText.style.color = color;
  }

  // ---------------------------------------------------------------------
  // Webcam + hand control
  // ---------------------------------------------------------------------

  async _setupWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
      this.videoElement.srcObject = stream;

      await new Promise((resolve) => {
        this.videoElement.onloadedmetadata = () => {
          this.videoElement.style.width = `${this.renderDiv.clientWidth}px`;
          this.videoElement.style.height = `${this.renderDiv.clientHeight}px`;
          resolve();
        };
      });
    } catch (error) {
      console.error('Error accessing webcam:', error);
      this._showError(`Webcam error: ${error.message}. Please allow camera access.`);
      throw error;
    }
  }

  async _setupBackgroundSegmenter() {
    if (this.backgroundSegmenterLoading || this.backgroundSegmenter) return;
    this.backgroundSegmenterLoading = true;
    this._updateBackgroundButton();
    try {
      const vision = await FilesetResolver.forVisionTasks(SEGMENTER_WASM_PATH);
      this.backgroundSegmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: SEGMENTER_MODEL_PATH, delegate: 'GPU' },
        runningMode: 'VIDEO',
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      });
    } catch (error) {
      console.warn('Person-aware blur unavailable; using full-camera blur fallback.', error);
      this.backgroundSegmenter = null;
      this.backgroundCanvas.style.display = 'none';
      if (this.backgroundBlurred) this.videoElement.style.filter = `blur(${BACKGROUND_BLUR_PX}px)`;
    } finally {
      this.backgroundSegmenterLoading = false;
      this._updateBackgroundButton();
    }
  }

  async _setupHandControl() {
    this.handControl = new HandControl({
      videoElement: this.videoElement,
      maxHands: 2,
      smoothing: 0.2,
      mirrorX: true,
    });

    this.handControl.on('rotate', ({ dx, dy }) => this._onRotate(dx, dy));
    this.handControl.on('zoom', ({ scale }) => this._onZoom(scale));
    this.handControl.on('pan', ({ dx, dy }) => this._onPan(dx, dy));
    this.handControl.on('pointer', ({ x, y }) => this._onPointer(x, y));
    this.handControl.on('select', ({ x, y }) => this._onSelect(x, y));
    this.handControl.on('sectionStart', () => this._onSectionStart());
    this.handControl.on('sectionMove', ({ delta }) => this._onSectionMove(delta));
    this.handControl.on('sectionEnd', () => this._onSectionEnd());
    this.handControl.on('pauseToggle', () => this._toggleGesturePause());
    this.handControl.on('resetView', () => this._resetView());
    this.handControl.on('landmarks', (hands) => this._drawLandmarks(hands));

    try {
      await this.handControl.start();
    } catch (error) {
      console.error('Error starting hand tracking:', error);
      this._showError(`Hand tracking error: ${error.message}.`);
      throw error;
    }
  }

  // ---------------------------------------------------------------------
  // Gesture -> scene wiring
  // ---------------------------------------------------------------------

  _onRotate(dx, dy) {
    if (!this.currentModel) return;
    this.currentModel.rotation.y -= dx * ROTATE_SENSITIVITY;
    this.currentModel.rotation.x -= dy * ROTATE_SENSITIVITY;
    this._rebuildSectionPlane();
    //this.audioManager.playInteractionClickSound();
  }

  _onZoom(scale) {
    if (!this.currentModel) return;
    const current = this.currentModel.scale.x;
    const next = Math.max(
      this.baseModelScale * ZOOM_MIN_FACTOR,
      Math.min(this.baseModelScale * ZOOM_MAX_FACTOR, current * scale)
    );
    this.currentModel.scale.set(next, next, next);
    //this.audioManager.playInteractionClickSound();
  }

  _onPan(dx, dy) {
    if (!this.currentModel) return;
    this.currentModel.position.x += dx * this.renderDiv.clientWidth * PAN_SENSITIVITY;
    this.currentModel.position.y -= dy * this.renderDiv.clientHeight * PAN_SENSITIVITY;
    this._rebuildSectionPlane();
  }

  _onPointer(x, y) {
    this.cursorDot.style.display = 'block';
    this.cursorDot.style.left = `${x * this.renderDiv.clientWidth}px`;
    this.cursorDot.style.top = `${y * this.renderDiv.clientHeight}px`;
  }

  _onSelect(x, y) {
    this.cursorDot.style.display = 'block';
    this.cursorDot.style.left = `${x * this.renderDiv.clientWidth}px`;
    this.cursorDot.style.top = `${y * this.renderDiv.clientHeight}px`;
    this.cursorDot.style.transform = 'scale(1.8)';
    setTimeout(() => {
      this.cursorDot.style.transform = 'scale(1)';
    }, 150);
    this.audioManager.playInteractionClickSound();
  }

  _onSectionStart() {
    this.sectionEnabled = true;
    this._updateSectionClipping();
    this.sectionBadge.style.backgroundColor = '#FFA500';
    this.sectionBadge.style.color = '#000000';
  }

  _onSectionMove(delta) {
    this.sectionOffset -= delta * this.renderDiv.clientHeight * SECTION_SENSITIVITY;
    this._rebuildSectionPlane();
  }

  _onSectionEnd() {
    this.sectionBadge.style.backgroundColor = 'white';
    this.sectionBadge.style.color = '#000000';
    // Section plane intentionally stays where it was left — no auto reset.
  }

  _updateSectionClipping() {
    if (!this.currentFragmentsModel?.object) return;
    this.currentFragmentsModel.object.traverse((child) => {
      if (!child.isMesh || !child.material) return;
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((material) => {
        material.clippingPlanes = this.sectionEnabled ? [this.sectionPlane] : [];
        material.needsUpdate = true;
      });
    });
  }

  _onKeyDown(event) {
    if (event.key === ' ') {
      event.preventDefault();
      this._toggleGesturePause();
    } else if (event.key.toLowerCase() === 'r') this._resetView();
    else if (event.key.toLowerCase() === 'b') this._toggleBackground();
    else if (event.key.toLowerCase() === 'l') this._toggleLandmarkOverlay();
    else if (event.key.toLowerCase() === 'i') this._toggleInfoPanel();
    else if (event.key === 'Escape' && this.infoPanelOpen) this._closeInfoPanel();
    else if (event.key === '1') this._setSectionOrientation(0);
    else if (event.key === '2') this._setSectionOrientation(1);
    else if (event.key === '3') this._setSectionOrientation(2);
  }

  _setSectionOrientation(index) {
    this.sectionOrientationIndex = index;
    this._rebuildSectionPlane();
    this._updateSectionBadgeText();
  }

  _rebuildSectionPlane() {
    const localNormal = SECTION_ORIENTATIONS[this.sectionOrientationIndex].normal;
    const modelPosition = this.currentModel?.position ?? new THREE.Vector3(0, 0, this.modelDefaultZ);
    const modelRotation = this.currentModel?.quaternion ?? new THREE.Quaternion();
    const normal = localNormal.clone().applyQuaternion(modelRotation).normalize();
    const planePoint = modelPosition.clone().addScaledVector(normal, this.sectionOffset);
    this.sectionPlane.setFromNormalAndCoplanarPoint(normal, planePoint);

    if (this.sectionPlaneVisual) {
      this.sectionPlaneVisual.position.copy(planePoint);
      this.sectionPlaneVisual.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
    }
  }

  _updateSectionBadgeText() {
    if (!this.sectionBadge) return;
    this.sectionBadge.innerText = `Section: ${SECTION_ORIENTATIONS[this.sectionOrientationIndex].label} (1/2/3)`;
  }

  // ---------------------------------------------------------------------
  // Main loop
  // ---------------------------------------------------------------------

  _animate() {
    requestAnimationFrame(this._animate.bind(this));
    this._updateGestureBadge();
    this._renderBackgroundBlur();
    this.renderer.render(this.scene, this.camera);
  }

  _updateGestureBadge() {
    if (!this.handControl) return;
    if (this.handControl.isPaused) {
      this.gestureBadge.innerText = 'Gestures paused';
      this.gestureBadge.style.backgroundColor = '#d9d9d9';
      this.pauseButton.innerText = 'Resume gestures';
      this.cursorDot.style.display = 'none';
      return;
    }
    const state = this.handControl.state;
    this.pauseButton.innerText = 'Pause gestures';
    if (this._lastBadgeState === state) return;
    this._lastBadgeState = state;
    this.gestureBadge.innerText = GESTURE_STATE_LABELS[state] || state;
    this.gestureBadge.title = `${state}: ${this._gestureHint(state)}`;
    this.gestureBadge.style.backgroundColor = GESTURE_STATE_COLORS[state] || '#888888';
    if (state !== GestureState.POINTING) this.cursorDot.style.display = 'none';
  }

  _gestureHint(state) {
    return {
      [GestureState.HAND_VISIBLE]: 'Make a clear pose to begin',
      [GestureState.POINTING]: 'Move your index finger to aim',
      [GestureState.PINCHING]: 'Move while pinching to rotate',
      [GestureState.ROTATING]: 'Release pinch to stop rotating',
      [GestureState.PANNING]: 'Move your open palm to pan',
      [GestureState.TWO_HAND_ZOOM]: 'Move hands apart or together',
      [GestureState.SECTION_ACTIVE]: 'Move or twist two raised fingers',
    }[state] || 'Show a hand to begin';
  }

  _toggleGesturePause() {
    if (!this.handControl) return;
    if (this.handControl.isPaused) this.handControl.resume();
    else this.handControl.pause();
    this._lastBadgeState = null;
    this._updateGestureBadge();
  }

  _resetView() {
    if (this.currentModel) {
      this.currentModel.position.set(0, 0, this.modelDefaultZ);
      this.currentModel.rotation.set(0, 0, 0);
      this.currentModel.scale.set(this.baseModelScale, this.baseModelScale, this.baseModelScale);
    }
    this.sectionOffset = 0;
    this.sectionEnabled = false;
    this._rebuildSectionPlane();
    this._updateSectionClipping();
    this.handControl?.reset();
    this._drawLandmarks([]);
    this._lastBadgeState = null;
    this.cursorDot.style.display = 'none';
    this._updateSectionBadgeText();
    this._updateGestureBadge();
  }

  _toggleBackground() {
    if (!this.backgroundVisible) {
      this.backgroundVisible = true;
      this.backgroundBlurred = false;
    } else if (!this.backgroundBlurred) {
      this.backgroundBlurred = true;
    } else {
      this.backgroundVisible = false;
      this.backgroundBlurred = false;
    }
    if (this.backgroundBlurred && !this.backgroundSegmenter) {
      this._setupBackgroundSegmenter();
    }
    this.videoElement.style.opacity = this.backgroundVisible ? '1' : '0';
    this.videoElement.style.filter = this.backgroundBlurred && !this.backgroundSegmenter
      ? `blur(${BACKGROUND_BLUR_PX}px)` : 'none';
    this.backgroundCanvas.style.display = this.backgroundBlurred ? 'block' : 'none';
    this._updateBackgroundButton();
  }

  _renderBackgroundBlur() {
    if (!this.backgroundBlurred || !this.backgroundSegmenter || !this.videoElement.videoWidth) return;
    const width = this.renderDiv.clientWidth;
    const height = this.renderDiv.clientHeight;
    if (this.backgroundCanvas.width !== width || this.backgroundCanvas.height !== height) {
      this.backgroundCanvas.width = width;
      this.backgroundCanvas.height = height;
      this.backgroundSourceCanvas.width = width;
      this.backgroundSourceCanvas.height = height;
    }
    const sourceContext = this.backgroundSourceCanvas.getContext('2d');
    const context = this.backgroundCanvas.getContext('2d');
    if (!sourceContext || !context) return;

    const crop = this._videoCropRect(width, height);
    sourceContext.clearRect(0, 0, width, height);
    this._drawMirroredVideo(sourceContext, width, height, crop);
    let result;
    try {
      result = this.backgroundSegmenter.segmentForVideo(this.videoElement, performance.now());
    } catch (error) {
      console.warn('Background blur frame failed; showing clear camera view.', error);
      this.backgroundBlurred = false;
      this.backgroundCanvas.style.display = 'none';
      this.videoElement.style.opacity = this.backgroundVisible ? '1' : '0';
      this.videoElement.style.filter = this.backgroundBlurred && !this.backgroundSegmenter
        ? `blur(${BACKGROUND_BLUR_PX}px)` : 'none';
      this._updateBackgroundButton();
      return;
    }
    const mask = result.categoryMask;
    if (!mask) return;
    const maskWidth = mask.width;
    const maskHeight = mask.height;
    if (this.backgroundMaskCanvas.width !== maskWidth || this.backgroundMaskCanvas.height !== maskHeight) {
      this.backgroundMaskCanvas.width = maskWidth;
      this.backgroundMaskCanvas.height = maskHeight;
    }
    const maskContext = this.backgroundMaskCanvas.getContext('2d');
    if (!maskContext) return;
    const values = mask.getAsUint8Array();
    const maskPixels = maskContext.createImageData(maskWidth, maskHeight);
    for (let index = 0; index < values.length; index++) {
      // selfie_segmenter's category mask is 0 = person/foreground, nonzero =
      // background — this alpha marks where to cut the "stay sharp" hole.
      const alpha = values[index] === 0 ? 255 : 0;
      const pixelIndex = index * 4;
      maskPixels.data[pixelIndex] = 255;
      maskPixels.data[pixelIndex + 1] = 255;
      maskPixels.data[pixelIndex + 2] = 255;
      maskPixels.data[pixelIndex + 3] = alpha;
    }
    maskContext.putImageData(maskPixels, 0, 0);

    context.clearRect(0, 0, width, height);
    context.filter = `blur(${BACKGROUND_BLUR_PX}px)`;
    this._drawMirroredVideo(context, width, height, crop);
    context.filter = 'none';
    context.globalCompositeOperation = 'destination-out';
    context.save();
    context.translate(width, 0);
    context.scale(-1, 1);
    context.drawImage(this.backgroundMaskCanvas, width - crop.x - crop.width, crop.y, crop.width, crop.height);
    context.restore();
    context.globalCompositeOperation = 'destination-over';
    context.drawImage(this.backgroundSourceCanvas, 0, 0, width, height);
    context.globalCompositeOperation = 'source-over';
    mask.close?.();
  }

  _videoCropRect(width, height) {
    const scale = Math.max(width / this.videoElement.videoWidth, height / this.videoElement.videoHeight);
    return {
      x: (width - this.videoElement.videoWidth * scale) / 2,
      y: (height - this.videoElement.videoHeight * scale) / 2,
      width: this.videoElement.videoWidth * scale,
      height: this.videoElement.videoHeight * scale,
    };
  }

  _drawMirroredVideo(context, width, height, crop) {
    context.save();
    context.translate(width, 0);
    context.scale(-1, 1);
    context.drawImage(this.videoElement, width - crop.x - crop.width, crop.y, crop.width, crop.height);
    context.restore();
  }

  _setupLandmarkOverlay() {
    this.landmarkOverlay = document.createElement('canvas');
    this.landmarkOverlay.className = 'landmark-overlay';
    this.landmarkOverlayVisible = true;
    this.renderDiv.appendChild(this.landmarkOverlay);
    this._resizeLandmarkOverlay();
  }

  _toggleLandmarkOverlay() {
    this.landmarkOverlayVisible = !this.landmarkOverlayVisible;
    this.landmarkOverlay.style.display = this.landmarkOverlayVisible ? 'block' : 'none';
    this._updateLandmarkButton();
  }

  _updateLandmarkButton() {
    if (!this.landmarkButton) return;
    this.landmarkButton.innerText = this.landmarkOverlayVisible ? 'Hide hands' : 'Show hands';
    this.landmarkButton.setAttribute('aria-pressed', String(this.landmarkOverlayVisible));
  }

  _drawLandmarks(hands) {
    const context = this.landmarkOverlay.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, this.landmarkOverlay.width, this.landmarkOverlay.height);
    if (!this.landmarkOverlayVisible) return;

    const connections = [
      [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
      [0, 13, 14, 15, 16], [0, 17, 18, 19, 20], [5, 9, 13, 17],
    ];
    context.lineWidth = 3;
    context.strokeStyle = '#00e5ff';
    context.fillStyle = '#ffffff';
    hands.forEach((landmarks) => {
      connections.forEach((chain) => {
        context.beginPath();
        chain.forEach((index, pointIndex) => {
          const point = landmarks[index];
          const { x, y } = this._landmarkToOverlay(point);
          if (pointIndex === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        context.stroke();
      });
      landmarks.forEach((point) => {
        const { x, y } = this._landmarkToOverlay(point);
        context.beginPath();
        context.arc(x, y, 4, 0, Math.PI * 2);
        context.fill();
      });
    });
  }

  _landmarkToOverlay(point) {
    const width = this.landmarkOverlay.width;
    const height = this.landmarkOverlay.height;
    const sourceWidth = this.videoElement.videoWidth || width;
    const sourceHeight = this.videoElement.videoHeight || height;
    const coverScale = Math.max(width / sourceWidth, height / sourceHeight);
    const renderedWidth = sourceWidth * coverScale;
    const renderedHeight = sourceHeight * coverScale;
    return {
      x: point.x * renderedWidth + (width - renderedWidth) / 2,
      y: point.y * renderedHeight + (height - renderedHeight) / 2,
    };
  }

  _resizeLandmarkOverlay() {
    if (!this.landmarkOverlay) return;
    this.landmarkOverlay.width = this.renderDiv.clientWidth;
    this.landmarkOverlay.height = this.renderDiv.clientHeight;
  }

  _updateBackgroundButton() {
    if (!this.backgroundButton) return;
    this.backgroundButton.innerText = this.backgroundSegmenterLoading
      ? 'Loading blur...'
      : !this.backgroundVisible
        ? 'Show background'
        : this.backgroundBlurred ? 'Blurred background' : 'Blur background';
    this.backgroundButton.setAttribute('aria-pressed', String(this.backgroundBlurred));
  }

  start() {
    this.renderDiv.addEventListener('click', () => {
      this.audioManager.resumeContext();
      if (this.gameState === 'error') {
        this._restartGame();
      }
    });
  }

  // ---------------------------------------------------------------------
  // Error / resize / restart
  // ---------------------------------------------------------------------

  _showError(message) {
    this._showStatusScreen(`ERROR: ${message}`, 'orange');
    this.gameState = 'error';
  }

  _restartGame() {
    this._updateStatusOverlayVisibility();
    this.gameState = 'tracking';
    this.clock.start();
  }

  _onResize() {
    const width = this.renderDiv.clientWidth;
    const height = this.renderDiv.clientHeight;
    this.camera.left = width / -2;
    this.camera.right = width / 2;
    this.camera.top = height / 2;
    this.camera.bottom = height / -2;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.videoElement.style.width = `${width}px`;
    this.videoElement.style.height = `${height}px`;
    this._resizeLandmarkOverlay();
  }
}
