const bcrypt = require('bcrypt');
const saltRounds = 12;
const myPlaintextPassword = 'SdvhvpKew4yuNXuP';


bcrypt.hashpw(myPlaintextPassword, saltRounds, function(err, hash) {
    console.log(hash)
});