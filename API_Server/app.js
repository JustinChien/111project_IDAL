const express = require('express');
const mysql = require('mysql');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const port = 3000;
const host_ip = '120.101.3.229'

// MySQL database connection
const connection = mysql.createConnection({
  host: 'localhost',
  // host: '118.169.128.154',
  user: 'Jotaro',
  password: 'jotarojoestar',
  database: 'project'
});

connection.connect();

// Middleware
app.use(bodyParser.json());
app.use(cors());

// API endpoint with exception catcher
app.get('/data', (req, res) => {
  try {
    let query = '';
    const queryType = req.query.queryType;

    switch (queryType) {
      case 'patient':
        query = 'SELECT * FROM patient';
        break;
      case 'byPID':
        const patientID = req.query.patientID;
        query = `SELECT * FROM report WHERE patientID = ${patientID}`;
        break;
      case 'byRID':
        const reportID = req.query.reportID;
        query = `SELECT * FROM report WHERE reportID IN (${reportID})`;
        break;
      default:
        throw new Error('Invalid query type parameter');
    }

    connection.query(query, (error, results) => {
      if (error) throw error;
      res.send(results);
    });
  } catch (error) {
    console.error(error);
    res.status(500).send('Internal server error:');
  }
});


// Start the server
app.listen(port, host_ip,() => {
  console.log(`Node.js Server _ API server`);
  console.log(`Server running on port ${port}`);
});
