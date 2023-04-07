const http = require('http');
const fs = require('fs');
const path = require('path');
const host_ip = '120.101.3.229';
const port = 25567;


const server = http.createServer((req, res) => {
  const reqUrl = req.url;

  // Check if the URL starts with '/images'
  if (reqUrl.startsWith('/images')) {
    // Remove the '/images' prefix from the URL
    const imagePath = reqUrl.slice('/images'.length);

    // Construct the full path to the image file
    const fullPath = path.join(__dirname, 'images', imagePath);

    // Check if the file exists
    fs.access(fullPath, fs.constants.R_OK, (err) => {
      if (err) {
        // File not found or permission denied
        res.statusCode = 404;
        res.end('Not found');
        return;
      }

      // Read the file and send it as the response
      fs.readFile(fullPath, (err, data) => {
        if (err) {
          res.statusCode = 500;
          res.end('Internal server error');
          return;
        }

        // Set the Content-Type header based on the file extension
        const ext = path.extname(fullPath).toLowerCase();
        const contentType = {
          '.jpg': 'image/jpeg',
          '.jpeg': 'image/jpeg',
          '.png': 'image/png',
          '.gif': 'image/gif',
        }[ext] || 'application/octet-stream';
        res.setHeader('Content-Type', contentType);

        res.end(data);
      });
    });
  } else {
    // URL does not start with '/images'
    res.statusCode = 404;
    res.end('Not found');
  }
});

server.listen(port, host_ip,() => {
  console.log(`Node.js Server _ Image server`);
  console.log('Server started on port 25567');
});
