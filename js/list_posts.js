const path = require('path');
const fs = require('fs');

const dir = path.join(__dirname, '../post-formatted');

fs.readdir(dir, (err, files) => {

    if (err) {
        return console.log('Unable to scan directory: ' + err);
    }

    files.forEach((file) => {
        console.log(file)
    })

})
