/**
 * Model Manager
 *
 * Handles model selection and loading.
 */

class ModelManager {
    constructor() {
        this.models = {
            junior: {
                path: 'models/easy.onnx',
                size: '~14 MB',
                description: '初级',
                params: '355万参数',
                training: '20480轮对弈 (温度1.0)',
                sizeBytes: 14 * 1024 * 1024
            },
            intermediate: {
                path: 'models/mid.onnx',
                size: '~14 MB',
                description: '中级',
                params: '355万参数',
                training: '20480轮对弈 (温度0.5)',
                sizeBytes: 14 * 1024 * 1024
            },
            advanced: {
                path: 'models/hard.onnx',
                size: '~14 MB',
                description: '高级',
                params: '355万参数',
                training: '20480轮对弈 (温度0.2)',
                sizeBytes: 14 * 1024 * 1024
            }
        };

        this.selectedModel = null;
    }

    /**
     * Initialize model manager.
     */
    initialize() {
        console.log('Model Manager initialized');
    }

    /**
     * Set selected model.
     * @param {string} modelType - 'junior', 'intermediate', or 'advanced'
     */
    setSelectedModel(modelType) {
        this.selectedModel = modelType;
    }

    /**
     * Get selected model path.
     * @returns {string} Path to ONNX model file
     */
    getModelPath() {
        return this.models[this.selectedModel].path;
    }

    /**
     * Load selected model.
     * @returns {Promise<OnnxAIPlayer>} Loaded AI player
     */
    async loadSelectedModel() {
        const modelPath = this.getModelPath();
        const aiPlayer = new OnnxAIPlayer(modelPath);
        await aiPlayer.loadModel();
        return aiPlayer;
    }
}
