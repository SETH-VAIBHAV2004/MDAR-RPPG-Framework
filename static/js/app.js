// Remote Photoplethysmography - Enhanced JavaScript Module
// Provides improved user experience with better error handling, loading states, and feedback

// Global state management
const AppState = {
    video: null,
    canvas: null,
    ctx: null,
    ws: null,
    isRunning: false,
    frameCount: 0,
    selectedCamera: null,
    reconnectAttempts: 0,
    maxReconnectAttempts: 3,
    reconnectTimeout: null,
    isUploading: false,
    lastHeartRate: null,
    heartRateHistory: [],
    maxHistoryLength: 50
};

// Configuration
const Config = {
    frameSkip: 3, // Send every 3rd frame to reduce bandwidth
    videoWidth: 640,
    videoHeight: 480,
    frameRate: 30,
    websocketReconnectDelay: 2000,
    uploadProgressUpdateInterval: 100,
    heartRateUpdateAnimationDuration: 300
};

// Utility functions
const Utils = {
    // Format time duration
    formatTime: (seconds) => {
        if (seconds < 60) return `${seconds.toFixed(1)}s`;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
    },

    // Format file size
    formatFileSize: (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    // Show notification
    showNotification: (message, type = 'info', duration = 5000) => {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(notification => notification.remove());

        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">&times;</button>
            </div>
        `;
        
        // Add styles dynamically
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            border-left: 4px solid ${type === 'success' ? '#4CAF50' : type === 'error' ? '#f44336' : type === 'warning' ? '#ff9800' : '#2196f3'};
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 10000;
            max-width: 400px;
            animation: slideInRight 0.3s ease-out;
        `;

        document.body.appendChild(notification);
        
        if (duration > 0) {
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.style.animation = 'slideOutRight 0.3s ease-in';
                    setTimeout(() => notification.remove(), 300);
                }
            }, duration);
        }
    }
};

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 15px;
    }
    
    .notification-close {
        background: none;
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
        padding: 0;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        transition: background-color 0.2s;
    }
    
    .notification-close:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    .pulse {
        animation: pulse 0.6s ease-in-out;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
`;
document.head.appendChild(style);

// Global functions for compatibility
window.startCamera = () => console.log('Camera start requested');
window.stopCamera = () => console.log('Camera stop requested');
window.refreshCameras = () => console.log('Camera refresh requested');

// Initialize application
function initializeApp() {
    Utils.showNotification('Remote Photoplethysmography system ready', 'success', 3000);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

/**
 * Enhanced MDAR rPPG Server - JavaScript Application
 * ================================================
 * 
 * This file provides interactive functionality for the web interface:
 * - File upload handling with progress indicators
 * - Real-time webcam processing with WebSocket
 * - Dynamic UI updates and animations
 * - Error handling and user feedback
 */

class rPPGApp {
    constructor() {
        this.websocket = null;
        this.isWebcamActive = false;
        this.video = null;
        this.canvas = null;
        this.context = null;
        this.frameCount = 0;
        
        this.initializeElements();
        this.bindEvents();
        this.showNotification('rPPG Application Loaded', 'info');
    }

    initializeElements() {
        // Get DOM elements
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.statusElement = document.getElementById('status');
        this.hrDisplay = document.getElementById('hrDisplay');
        this.confidenceDisplay = document.getElementById('confidence');
        
        // Initialize canvas context if canvas exists
        if (this.canvas) {
            this.context = this.canvas.getContext('2d');
            this.canvas.width = 640;
            this.canvas.height = 480;
        }
    }

    bindEvents() {
        // File upload handlers
        document.querySelectorAll('.upload-form').forEach(form => {
            form.addEventListener('submit', (e) => this.handleFileUpload(e));
        });

        // Webcam control buttons
        const startCameraBtn = document.getElementById('startCamera');
        const stopCameraBtn = document.getElementById('stopCamera');
        
        if (startCameraBtn) {
            startCameraBtn.addEventListener('click', () => this.startWebcam());
        }
        
        if (stopCameraBtn) {
            stopCameraBtn.addEventListener('click', () => this.stopWebcam());
        }

        // Native webcam buttons
        const nativeWebcamBtn = document.getElementById('nativeWebcamBtn');
        if (nativeWebcamBtn) {
            nativeWebcamBtn.addEventListener('click', () => this.startNativeWebcam());
        }

        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.websocket) {
                console.log('Page hidden, pausing WebSocket');
            } else if (!document.hidden && this.websocket) {
                console.log('Page visible, resuming WebSocket');
            }
        });
    }

    // File Upload Handling
    async handleFileUpload(event) {
        event.preventDefault();
        
        const form = event.target;
        const fileInput = form.querySelector('input[type="file"]');
        let outputElement = form.querySelector('.output') || document.getElementById('videoResults');
        const submitButton = form.querySelector('button[type="submit"]');
        let statusElement = document.getElementById('videoUploadStatus');
        
        if (!fileInput.files.length) {
            this.showNotification('Please select a video file first', 'error');
            return;
        }

        const file = fileInput.files[0];
        
        // Validate video file
        if (!this.isValidVideoFile(file)) {
            this.showNotification('Please select a valid video file (.mp4, .avi, .mov, .mkv, .webm, .m4v)', 'error');
            return;
        }
        
        // Check file size (limit to 2GB)
        const maxSize = 2 * 1024 * 1024 * 1024; // 2GB
        if (file.size > maxSize) {
            this.showNotification('Video file is too large. Maximum size is 2GB.', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        // Update UI
        if (statusElement) {
            statusElement.style.display = 'block';
            statusElement.textContent = `Processing ${file.name}... (${this.formatFileSize(file.size)})`;
            statusElement.className = 'status info';
        }
        
        outputElement.innerHTML = this.createLoadingHTML(`Analyzing video for heart rate...`);
        
        const originalButtonText = submitButton.innerHTML;
        submitButton.disabled = true;
        submitButton.innerHTML = '<div class="spinner"></div> Processing Video...';

        try {
            const endpoint = form.action || '/api/video/upload';
            const startTime = Date.now();
            
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            let result;
            const contentType = response.headers.get('content-type');
            
            if (contentType && contentType.includes('application/json')) {
                result = await response.json();
            } else {
                result = { error: await response.text() };
            }

            const processingTime = (Date.now() - startTime) / 1000;

            if (response.ok) {
                this.displayVideoUploadResult(outputElement, result, file.name, processingTime);
                this.showNotification(`Video processed successfully in ${processingTime.toFixed(1)}s`, 'success');
                
                if (statusElement) {
                    statusElement.textContent = `✅ Video processed successfully`;
                    statusElement.className = 'status success';
                }
            } else {
                throw new Error(result.error || result.detail || 'Video processing failed');
            }

        } catch (error) {
            console.error('Video upload error:', error);
            outputElement.innerHTML = this.createErrorHTML(error.message);
            this.showNotification(`Video processing failed: ${error.message}`, 'error');
            
            if (statusElement) {
                statusElement.textContent = `❌ Processing failed: ${error.message}`;
                statusElement.className = 'status error';
            }
        } finally {
            // Reset button
            submitButton.disabled = false;
            submitButton.innerHTML = originalButtonText;
        }
    }
    
    isValidVideoFile(file) {
        const validTypes = [
            'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo',
            'video/x-matroska', 'video/webm', 'video/x-m4v'
        ];
        const validExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'];
        
        const hasValidType = validTypes.includes(file.type);
        const hasValidExtension = validExtensions.some(ext => 
            file.name.toLowerCase().endsWith(ext)
        );
        
        return hasValidType || hasValidExtension;
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    displayVideoUploadResult(outputElement, result, fileName, processingTime) {
        let html = `
            <div class="result-container card">
                <h3>🎬 Video Analysis Results</h3>
                <h4>📄 ${fileName}</h4>
                <div class="metrics-grid">
        `;

        if (result.hr_bpm !== undefined) {
            const reliable = result.reliable ? '✅' : '⚠️';
            const reliableText = result.reliable ? 'Reliable' : 'May be unreliable';
            
            html += `
                <div class="metric-card">
                    <div class="metric-value hr-display" style="font-size: 2.5rem; margin: 10px 0;">${result.hr_bpm.toFixed(1)}</div>
                    <div class="metric-label">Heart Rate (BPM)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${result.confidence.toFixed(1)}%</div>
                    <div class="metric-label">Confidence</div>
                </div>
            `;
        }
        
        html += `</div>`;
        
        // Status indicator
        if (result.hr_bpm !== undefined) {
            const reliable = result.reliable ? '✅' : '⚠️';
            const reliableText = result.reliable ? 'Reliable reading' : 'Reading may be unreliable';
            html += `
                <div class="status ${result.reliable ? 'success' : 'warning'}">
                    ${reliable} ${reliableText}
                </div>
            `;
        }

        // Method comparison
        if (result.hr_mdar !== undefined || result.hr_pos !== undefined || result.hr_chrom !== undefined) {
            html += `
                <div class="method-comparison">
                    <h4>📊 Method Comparison</h4>
                    ${result.hr_mdar ? `
                    <div class="method-row">
                        <span class="method-name">MDAR Model:</span>
                        <span class="method-value">${result.hr_mdar.toFixed(1)} BPM</span>
                    </div>
                    ` : ''}
                    ${result.hr_pos ? `
                    <div class="method-row">
                        <span class="method-name">POS Method:</span>
                        <span class="method-value">${result.hr_pos.toFixed(1)} BPM</span>
                    </div>
                    ` : ''}
                    ${result.hr_chrom ? `
                    <div class="method-row">
                        <span class="method-name">CHROM Method:</span>
                        <span class="method-value">${result.hr_chrom.toFixed(1)} BPM</span>
                    </div>
                    ` : ''}
                </div>
            `;
        }

        // Processing details
        html += `
            <div class="method-comparison">
                <h4>🔧 Processing Details</h4>
                ${result.fps ? `
                <div class="method-row">
                    <span class="method-name">Frame Rate:</span>
                    <span class="method-value">${result.fps.toFixed(1)} fps</span>
                </div>
                ` : ''}
                ${result.frames_processed !== undefined ? `
                <div class="method-row">
                    <span class="method-name">Frames Processed:</span>
                    <span class="method-value">${result.frames_processed}</span>
                </div>
                ` : ''}
                ${result.motion_rejects !== undefined ? `
                <div class="method-row">
                    <span class="method-name">Motion Rejects:</span>
                    <span class="method-value">${result.motion_rejects}</span>
                </div>
                ` : ''}
                ${result.waveform_length !== undefined ? `
                <div class="method-row">
                    <span class="method-name">Waveform Length:</span>
                    <span class="method-value">${result.waveform_length} samples</span>
                </div>
                ` : ''}
                ${processingTime ? `
                <div class="method-row">
                    <span class="method-name">Processing Time:</span>
                    <span class="method-value">${processingTime.toFixed(1)}s</span>
                </div>
                ` : ''}
            </div>
        `;

        html += '</div>';
        outputElement.innerHTML = html;
    }
    
    displayUploadResult(outputElement, result, fileName) {
        // Legacy function for backward compatibility
        this.displayVideoUploadResult(outputElement, result, fileName, null);
    }

    // Webcam Functionality
    async startWebcam() {
        try {
            this.showNotification('Starting webcam...', 'info');
            
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: 640, 
                    height: 480, 
                    frameRate: 30,
                    facingMode: 'user'
                }
            });

            this.video.srcObject = stream;
            await new Promise(resolve => {
                this.video.onloadedmetadata = resolve;
            });

            // Initialize WebSocket connection
            await this.connectWebSocket();
            
            this.isWebcamActive = true;
            this.updateWebcamUI(true);
            this.startFrameCapture();
            
            this.showNotification('Webcam started successfully', 'success');

        } catch (error) {
            console.error('Webcam error:', error);
            this.showNotification(`Webcam failed: ${error.message}`, 'error');
            this.updateStatus('❌ Camera access denied', 'error');
        }
    }

    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/live`;
            
            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                resolve();
            };

            this.websocket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                if (message.type === 'prediction') {
                    this.updatePredictionDisplay(message.data);
                }
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showNotification('WebSocket connection failed', 'error');
                reject(error);
            };

            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.websocket = null;
            };

            // Connection timeout
            setTimeout(() => {
                if (this.websocket && this.websocket.readyState !== WebSocket.OPEN) {
                    reject(new Error('WebSocket connection timeout'));
                }
            }, 5000);
        });
    }

    startFrameCapture() {
        if (!this.isWebcamActive || !this.websocket) return;

        this.frameCount++;
        
        // Send every 3rd frame to reduce bandwidth
        if (this.frameCount % 3 === 0) {
            this.context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            const frameData = this.canvas.toDataURL('image/jpeg', 0.7);

            if (this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'frame',
                    frame: frameData
                }));
            }
        }

        requestAnimationFrame(() => this.startFrameCapture());
    }

    stopWebcam() {
        this.isWebcamActive = false;
        
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
        }

        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }

        this.updateWebcamUI(false);
        this.resetPredictionDisplay();
        this.showNotification('Webcam stopped', 'info');
    }

    updateWebcamUI(isActive) {
        const startBtn = document.getElementById('startCamera');
        const stopBtn = document.getElementById('stopCamera');
        
        if (startBtn && stopBtn) {
            startBtn.disabled = isActive;
            stopBtn.disabled = !isActive;
        }
    }

    updatePredictionDisplay(data) {
        if (data.hr_bpm && this.hrDisplay) {
            this.hrDisplay.textContent = `${data.hr_bpm.toFixed(1)} BPM`;
            this.hrDisplay.classList.add('pulse-animation');
            setTimeout(() => {
                this.hrDisplay.classList.remove('pulse-animation');
            }, 1000);
        }

        if (data.confidence !== undefined && this.confidenceDisplay) {
            this.confidenceDisplay.textContent = `Confidence: ${data.confidence.toFixed(1)}%`;
        }

        // Update method comparison
        const elements = {
            hrMdar: data.hr_mdar,
            hrPos: data.hr_pos,
            hrChrom: data.hr_chrom,
            bufferSize: data.buffer_size
        };

        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element && value !== undefined) {
                element.textContent = typeof value === 'number' ? value.toFixed(1) : value;
            }
        });

        // Update status
        if (data.error) {
            this.updateStatus(`❌ Error: ${data.error}`, 'error');
        } else if (data.waiting_for_data) {
            this.updateStatus(`🔄 Collecting data... (${data.buffer_size}/90 frames)`, 'warning');
        } else if (data.hr_bpm) {
            const reliable = data.reliable ? '✅ Reading reliable' : '⚠️ Reading may be unreliable';
            const statusClass = data.reliable ? 'success' : 'warning';
            this.updateStatus(reliable, statusClass);
        }
    }

    resetPredictionDisplay() {
        if (this.hrDisplay) this.hrDisplay.textContent = '-- BPM';
        if (this.confidenceDisplay) this.confidenceDisplay.textContent = 'Confidence: --%';
        this.updateStatus('Camera stopped', '');

        // Reset method values
        ['hrMdar', 'hrPos', 'hrChrom', 'bufferSize'].forEach(id => {
            const element = document.getElementById(id);
            if (element) element.textContent = '--';
        });
    }

    // Native Webcam Integration
    async startNativeWebcam() {
        try {
            this.showNotification('Starting native webcam...', 'info');
            this.updateStatus('Starting native webcam...', 'info');

            const response = await fetch('/api/webcam/start', {
                method: 'POST'
            });

            const result = await response.json();

            if (response.ok) {
                this.updateStatus(result.message, 'success');
                this.showNotification('Native webcam started', 'success');
            } else {
                throw new Error(result.detail || result.message);
            }

        } catch (error) {
            console.error('Native webcam error:', error);
            this.updateStatus(`Error: ${error.message}`, 'error');
            this.showNotification(`Native webcam failed: ${error.message}`, 'error');
        }
    }

    // UI Utility Methods
    updateStatus(message, className = '') {
        if (this.statusElement) {
            this.statusElement.textContent = message;
            this.statusElement.className = `status ${className}`;
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${this.getNotificationIcon(type)}</span>
                <span class="notification-message">${message}</span>
            </div>
        `;

        // Add styles if not already present
        this.ensureNotificationStyles();

        // Add to document
        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => notification.classList.add('show'), 100);

        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }

    getNotificationIcon(type) {
        const icons = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        };
        return icons[type] || 'ℹ️';
    }

    ensureNotificationStyles() {
        if (document.getElementById('notification-styles')) return;

        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: var(--glass-bg);
                border: 1px solid var(--glass-border);
                border-radius: var(--border-radius);
                padding: 15px 20px;
                backdrop-filter: blur(15px);
                -webkit-backdrop-filter: blur(15px);
                box-shadow: var(--shadow-medium);
                z-index: 10000;
                transform: translateX(400px);
                opacity: 0;
                transition: all 0.3s ease;
                max-width: 350px;
            }

            .notification.show {
                transform: translateX(0);
                opacity: 1;
            }

            .notification-content {
                display: flex;
                align-items: center;
                gap: 10px;
                color: var(--text-primary);
                font-weight: 500;
            }

            .notification-icon {
                font-size: 1.2rem;
            }

            .notification-success {
                border-color: var(--success-color);
            }

            .notification-error {
                border-color: var(--error-color);
            }

            .notification-warning {
                border-color: var(--warning-color);
            }

            .notification-info {
                border-color: var(--info-color);
            }
        `;
        document.head.appendChild(style);
    }

    // Helper methods for HTML generation
    createLoadingHTML(message) {
        return `
            <div class="loading">
                <div class="spinner"></div>
                <span>${message}</span>
            </div>
        `;
    }

    createErrorHTML(message) {
        return `
            <div class="status error">
                <span>❌ ${message}</span>
            </div>
        `;
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.rppgApp = new rPPGApp();
});

// Add pulse animation for heart rate display
const style = document.createElement('style');
style.textContent = `
    .pulse-animation {
        animation: heartbeat 0.5s ease-in-out;
    }

    @keyframes heartbeat {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
`;
document.head.appendChild(style);
