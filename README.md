# 國立宜蘭大學 資工系 IDAL實驗室 111學年度 專題

### 專題題目 : 基於深度學習與混合實境介面之乳癌辨識輔助診斷系統
### Breast Cancer Diagnostic Assistant System Using Deep Learning and Mixed Reality Interface

團隊成員 : 簡名駿 B0943017, 王昱捷 B0943022, 郭又萓 B0943033, 張光猛 B0943048    
指導教授 : 林斯寅 教授  

根據WHO的統計每年罹患乳癌的人數不斷增加，我們希望能夠使用所學來解決這項問題。  
因此在教授安排下與陽大陳慶耀醫師討論後整理出了三項功能: 即時良惡性預測功能、病例比較功能、病例查詢功能。

即時良惡性預測功能 : 使用CBIS-DDSM中的裁切圖做為訓練用資料、經過前處理後使用EfficientNet B7做遷移學習。  
病例比較功能 : 使用OpenCV套件實作，仍在改善中。  
病例查詢功能 : 從資料庫取得資料後在Hololens上呈現。  


## Key Features

### Patient Record Query
- Facial recognition based patient identification and data retrieval
- Historical mammogram and report browsing
- Integration with MariaDB for patient data storage
- Support for viewing comprehensive medical history

### Diagnostic Image Analysis
- Mammogram comparison across different time periods
- Automatic difference detection and marking
- ROI cropping functionality for detailed examination
- Real-time AI prediction for benign/malignant classification
- Detailed region measurements and visualization

### Mixed Reality Interface (HoloLens 2)
- Hands-free gesture and voice command control
- Dynamic image sizing and positioning
- Remote access to patient data and images
- Real-time diagnostic feedback display
- Portable workspace for flexible use

## System Architecture

### Frontend (HoloLens 2)
- Unity with MRTK integration
- WebGL-based server communication
- Real-time image processing
- Interactive 3D user interface

### Backend
- Node.js server for data management and API services
- Flask server for image processing and AI predictions
- MariaDB for patient records and reporting
- File system for image storage and management

### Core Services
- Patient data retrieval and management
- Image comparison and analysis
- Deep learning based predictions
- Real-time data synchronization

## Future Improvements

### Model Enhancement
- Improve prediction accuracy through advanced preprocessing
- Implement automated ROI generation
- Integrate additional imaging types
- Enhance feature detection capabilities

### System Optimization
- Improve real-time processing speed
- Enhance user interface based on doctor feedback
- Develop EMR integration capabilities
- Strengthen security measures

### Clinical Integration
- Expand hospital workflow support
- Implement multi-department collaboration features
- Add support for different diagnostic scenarios
- Create comprehensive audit system
