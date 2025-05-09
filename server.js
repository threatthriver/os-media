const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const dotenv = require('dotenv');
// const auth = require('./middleware/auth');

// Load environment variables
dotenv.config();

// Initialize Express
const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Database connection
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('MongoDB connection error:', err));

// Basic route
app.get('/', (req, res) => {
  res.send('Social Media API');
});

// Post routes
const Post = require('./models/Post');
const auth = require('./middleware/auth');

// Create post
app.post('/api/posts', auth, async (req, res) => {
  try {
    const { title, content } = req.body;
    
    // Input validation
    if (!title || !content) {
      return res.status(400).json({
        success: false,
        error: 'Title and content are required',
        code: 'POST_MISSING_FIELDS'
      });
    }
    
    if (title.length > 100) {
      return res.status(400).json({
        success: false,
        error: 'Title must be less than 100 characters',
        code: 'POST_TITLE_TOO_LONG'
      });
    }
    
    const post = new Post({
      title,
      content,
      user: req.user.id
    });
    
    await post.save();
    
    res.status(201).json({
      success: true,
      data: post,
      message: 'Post created successfully'
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      code: 'SERVER_ERROR'
    });
  }
});

// Get all posts with pagination
app.get('/api/posts', async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const skip = (page - 1) * limit;
    
    const posts = await Post.find()
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit)
      .lean();
      
    const totalPosts = await Post.countDocuments();
    
    res.json({
      success: true,
      data: posts,
      pagination: {
        page,
        limit,
        total: totalPosts,
        pages: Math.ceil(totalPosts / limit)
      }
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      code: 'SERVER_ERROR'
    });
  }
});

// Update post
app.put('/api/posts/:id', auth, async (req, res) => {
  try {
    const { title, content } = req.body;
    const post = await Post.findById(req.params.id);
    
    if (!post) {
      return res.status(404).json({
        success: false,
        error: 'Post not found',
        code: 'POST_NOT_FOUND'
      });
    }
    
    // Check if user owns the post
    if (post.user.toString() !== req.user.id) {
      return res.status(401).json({
        success: false,
        error: 'Not authorized to update this post',
        code: 'NOT_AUTHORIZED'
      });
    }
    
    // Input validation
    if (!title || !content) {
      return res.status(400).json({
        success: false,
        error: 'Title and content are required',
        code: 'POST_MISSING_FIELDS'
      });
    }
    
    if (title.length > 100) {
      return res.status(400).json({
        success: false,
        error: 'Title must be less than 100 characters',
        code: 'POST_TITLE_TOO_LONG'
      });
    }
    
    post.title = title;
    post.content = content;
    await post.save();
    
    res.json({
      success: true,
      data: post,
      message: 'Post updated successfully'
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      code: 'SERVER_ERROR'
    });
  }
});

// Like post
app.post('/api/posts/:id/like', auth, async (req, res) => {
  try {
    const post = await Post.findById(req.params.id);
    
    if (!post) {
      return res.status(404).json({
        success: false,
        error: 'Post not found',
        code: 'POST_NOT_FOUND'
      });
    }
    
    // Check if user already liked the post
    if (post.likes.some(like => like.user.toString() === req.user.id)) {
      return res.status(400).json({
        success: false,
        error: 'Post already liked',
        code: 'POST_ALREADY_LIKED'
      });
    }
    
    post.likes.unshift({ user: req.user.id });
    await post.save();
    
    res.json({
      success: true,
      data: post,
      message: 'Post liked successfully'
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      code: 'SERVER_ERROR'
    });
  }
});

// Comment on post
app.post('/api/posts/:id/comment', auth, async (req, res) => {
  try {
    const { text } = req.body;
    const post = await Post.findById(req.params.id);
    
    if (!post) {
      return res.status(404).json({
        success: false,
        error: 'Post not found',
        code: 'POST_NOT_FOUND'
      });
    }
    
    if (!text) {
      return res.status(400).json({
        success: false,
        error: 'Comment text is required',
        code: 'COMMENT_MISSING_TEXT'
      });
    }
    
    const newComment = {
      user: req.user.id,
      text
    };
    
    post.comments.unshift(newComment);
    await post.save();
    
    res.status(201).json({
      success: true,
      data: post,
      message: 'Comment added successfully'
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      code: 'SERVER_ERROR'
    });
  }
});

// Start server
const PORT = process.env.PORT || 5003;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});