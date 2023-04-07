const express = require('express');
const axios = require('axios');

const app = express();
const PORT = 25566;
const FLASK_PORT = 5000;
const FLASK_URL = `http://120.101.3.229:${FLASK_PORT}/predict`;

app.use(express.json());

app.post('/predict', async (req, res) => {
  const { image } = req.body;

  try {
    const response = await axios.post(FLASK_URL, { image });
    const prediction = response.data.prediction;
    res.send(`The model predicts: ${prediction}`);
  } catch (error) {
    console.error(error);
    res.status(500).send('Error making prediction');
  }
});


const compareRouter = express.Router();
compareRouter.post('/compare_image', async (req, res) => {
  const { id1, id2, img_type } = req.body;
  try {
    // send those parameters to flask and get the result
    const response = await axios.post(FLASK_URL,{id1, id2, img_type});
    const image = response.data.image

    // Send the image as a response
    res.writeHead(200, {'Content-Type': 'image/png' }); //image should already encoded into base64
    res.send(image);
  } catch (error) {
    console.error(error);
    res.status(500).send('Error comparing images');
  }
});

compareRouter.post('/generate_cropped', async (req, res) => {
  const { id, img_type } = req.body;
  try {
    const response = await axios.post(FLASK_URL,{ id, img_type});
    const images = response.data.images

    if (Array.isArray(images)) {
      // If images is an array, send each image as a separate response
      images.forEach(image => {
        res.writeHead(200, {'Content-Type': 'image/png' }); //image should already encoded into base64
        res.write(image);
      });
      res.end();
    } else {
      // If images is a single image, send it as a response
      res.writeHead(200, {'Content-Type': 'image/png' }); //image should already encoded into base64
      res.send(images);
    }
  } catch (error) {
    console.error(error);
    res.status(500).send('Error generating cropped images');
  }
});

compareRouter.post('/marking_abnormal', async (req, res) => {
  const { id, img_type } = req.body;
  try {
    // send those parameters to flask and get the result
    const response = await axios.post(FLASK_URL,{id, img_type});
    const image = response.data.image

    // Send the image as a response
    res.writeHead(200, {'Content-Type': 'image/png' }); //image should already encoded into base64
    res.send(image);
  } catch (error) {
    console.error(error);
    res.status(500).send('Error marking abnormal parts');
  }
});

app.use('/compare',compareRouter);

app.post('/facenet', async (req, res) => {
  const { image } = req.body;

  try {
    const response = await axios.post(FLASK_URL, { image });
    const PID = response.data.PID;
    res.send(`patientID : ${PID}`);
  } catch (error) {
    console.error(error);
    res.status(500).send('Error recognizing face');
  }
});

app.listen(PORT, () => {
  console.log(`Node.js Server _ Flask server`);
  console.log(`Server listening on port ${PORT}`);
});
