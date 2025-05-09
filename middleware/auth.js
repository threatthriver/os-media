const jwt = require('jsonwebtoken');
const User = require('../models/User');
const rateLimit = require('express-rate-limit');

// Rate limiting for auth requests
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later'
});

module.exports = async function(req, res, next) {
  // Apply rate limiting
  await new Promise((resolve, reject) => {
    authLimiter(req, res, (err) => {
      if (err) return reject(err);
      resolve();
    });
  });
  // Get token from header
  const token = req.header('Authorization')?.replace('Bearer ', '');

  // Check if no token
  if (!token) {
    return res.status(401).json({ 
      success: false,
      error: 'No token provided',
      code: 'AUTH_NO_TOKEN'
    });
  }

  // Verify token
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET, {
      algorithms: ['HS256'],
      ignoreExpiration: false
    });
    
    req.user = await User.findById(decoded.user.id).select('-password');
    if (!req.user) {
      return res.status(401).json({
        success: false,
        error: 'User not found',
        code: 'AUTH_USER_NOT_FOUND'
      });
    }
    
    next();
  } catch (err) {
    let errorMessage = 'Invalid token';
    let errorCode = 'AUTH_INVALID_TOKEN';
    
    if (err.name === 'TokenExpiredError') {
      errorMessage = 'Token expired';
      errorCode = 'AUTH_TOKEN_EXPIRED';
    } else if (err.name === 'JsonWebTokenError') {
      errorMessage = 'Malformed token';
      errorCode = 'AUTH_MALFORMED_TOKEN';
    }
    
    res.status(401).json({
      success: false,
      error: errorMessage,
      code: errorCode
    });
  }
};