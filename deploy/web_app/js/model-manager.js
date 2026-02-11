/**
 * Model Manager
 *
 * Handles model selection and loading.
 */

class ModelManager {
    constructor() {
        this.models = {
            junior: {
                path: 'models/dial.onnx',
                temperature: 1.0,
            },
            intermediate: {
                path: 'models/cello.onnx',
                temperature: 0.7,
            },
            advanced: {
                path: 'models/melody.onnx',
                temperature: 0.5,
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
     * Get selected model temperature.
     * @returns {number} Temperature for softmax sampling
     */
    getModelTemperature() {
        return this.models[this.selectedModel].temperature;
    }

    /**
     * Load selected model.
     * @returns {Promise<OnnxAIPlayer>} Loaded AI player
     */
    async loadSelectedModel() {
        const modelPath = this.getModelPath();
        const temperature = this.getModelTemperature();
        const aiPlayer = new OnnxAIPlayer(modelPath, temperature);
        await aiPlayer.loadModel();
        return aiPlayer;
    }
}
