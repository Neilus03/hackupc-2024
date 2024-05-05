# <div align="center">Inditech: Multimodal AI Shopping App for ZARA

![image](https://github.com/Neilus03/hackupc-2024/assets/127413352/62cca9e3-d9fc-4cb2-a87d-c0c1fb452101)


## Overview

Inditech is designed to revolutionize the online shopping experience at ZARA by integrating advanced multimodal AI capabilities. This application allows users to interactively search and navigate through ZARA's product catalog using voice commands and hand gestures. Our goal is to simplify and enhance the shopping process, making it more interactive and user-friendly.

### Inspiration

Our goal was to integrate multimodal AI capabilities to enhance the e-commerce experience for ZARA. We recognized the potential of voice and gesture-based navigation to simplify and enrich online shopping, making it more interactive and user-friendly.

### Features

- **Voice Search**: Users can speak to the app to find products, using natural language processing to analyze and respond to user queries.
- **Gesture Navigation**: The app includes a hand gesture detection algorithm that allows users to give thumbs up or thumbs down to navigate through product options.
- **Multiple Product Views**: Clicking on products displays them from various angles, enhancing the virtual shopping experience.

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **AI & Machine Learning**: OpenCV, Hugging Face Transformers, Fashion-CLIP, OpenAI Whisper

## Installation

To get started with Inditech, clone the repository and follow these installation steps:

```bash
git clone https://github.com/Neilus03/hackupc-2024.git
cd hackupc-2024
# Install dependencies
pip install -r requirements.txt
# Run the application
python app.py
```

## Usage

After launching the app, navigate to `localhost:5000` in your web browser to start using Inditech. Use your microphone for voice commands and your webcam for gesture recognition.

## Challenges We Ran Into

During development, we encountered issues such as library compatibility conflicts and difficulties in implementing the YOLO model for gesture detection, especially in varying lighting conditions. Maintaining a stable connection with our web hosting service also posed a significant challenge.

## Accomplishments

Despite these challenges, Inditech is functional and effective in understanding user inputs through both voice and gestures. The technology has significant potential for adoption by retail giants like ZARA.

## Lessons Learned

This project deepened our understanding of applying multimodal AI in real-world applications, enhancing our debugging skills and teamwork in a dynamic setting.

## Future Plans

We aim to refine the gesture recognition accuracy and pilot this system with ZARA to evaluate its impact on consumer engagement and sales.

## Contributing

We welcome contributions! Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

- **Neil de la Fuente** - [Github](https://github.com/Neilus03) - [LinkedIn](https://www.linkedin.com/in/neil-de-la-fuente/)
- **Maria Pilligua** - [Github](https://github.com/mpilligua) - [LinkedIn](https://www.linkedin.com/in/mariapilligua/)
- **Daniel Vidal** - [Github](https://github.com/Dani13vg) - [LinkedIn](https://www.linkedin.com/in/daniel-alejandro-vidal-guerra-21386b266/)
- **Alex Roldan** - [Github](https://github.com/alrocb) - [LinkedIn](https://www.linkedin.com/in/alex-roldan-55488a215/)

## Acknowledgments

- Thanks to HackUPC 2024 for the opportunity to develop this innovative solution.
- Thanks to Inditex for proposing the challenge and providing the data.
- Special thanks to mentors and everyone who supported us during the hackathon.


