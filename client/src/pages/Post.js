import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Container, TextField, Button, Typography, Box, 
  Card, CardContent, IconButton, Avatar, Divider 
} from '@mui/material';
import FavoriteIcon from '@mui/icons-material/Favorite';
import CommentIcon from '@mui/icons-material/Comment';
import axios from 'axios';

const Post = () => {
  const [formData, setFormData] = useState({
    title: '',
    content: ''
  });
  const [posts, setPosts] = useState([]);
  const [commentText, setCommentText] = useState('');
  const [activeCommentPost, setActiveCommentPost] = useState(null);
  const navigate = useNavigate();
  
  useEffect(() => {
    fetchPosts();
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const token = localStorage.getItem('token');
      await axios.post('/api/posts', formData, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      fetchPosts();
      setFormData({ title: '', content: '' });
    } catch (err) {
      console.error(err);
    }
  };

  const fetchPosts = async () => {
    try {
      const res = await axios.get('/api/posts');
      setPosts(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Create Post
        </Typography>
        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            margin="normal"
            label="Title"
            name="title"
            value={formData.title}
            onChange={handleChange}
            required
          />
          <TextField
            fullWidth
            margin="normal"
            label="Content"
            name="content"
            multiline
            rows={4}
            value={formData.content}
            onChange={handleChange}
            required
          />
          <Button
            type="submit"
            variant="contained"
            color="primary"
            fullWidth
            sx={{ mt: 3, mb: 2 }}
          >
            Post
          </Button>
        </form>
        
        <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>
          Your Posts
        </Typography>
        {posts.map((post) => (
          <Card key={post._id} sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6">{post.title}</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                {post.content}
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <IconButton onClick={() => handleLike(post._id)}>
                  <FavoriteIcon color={post.likes?.some(like => like.user === localStorage.getItem('userId')) ? 'error' : 'inherit'} />
                  <Typography variant="caption" sx={{ ml: 0.5 }}>
                    {post.likes?.length || 0}
                  </Typography>
                </IconButton>
                
                <IconButton onClick={() => setActiveCommentPost(activeCommentPost === post._id ? null : post._id)}>
                  <CommentIcon />
                  <Typography variant="caption" sx={{ ml: 0.5 }}>
                    {post.comments?.length || 0}
                  </Typography>
                </IconButton>
              </Box>
              
              {activeCommentPost === post._id && (
                <Box sx={{ mt: 2 }}>
                  <TextField
                    fullWidth
                    size="small"
                    placeholder="Add a comment"
                    value={commentText}
                    onChange={(e) => setCommentText(e.target.value)}
                  />
                  <Button 
                    size="small" 
                    variant="contained" 
                    sx={{ mt: 1 }}
                    onClick={() => handleComment(post._id)}
                  >
                    Post Comment
                  </Button>
                  
                  {post.comments?.map((comment, index) => (
                    <Box key={index} sx={{ mt: 2 }}>
                      <Divider sx={{ mb: 1 }} />
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Avatar sx={{ width: 24, height: 24 }} />
                        <Typography variant="body2">{comment.text}</Typography>
                      </Box>
                    </Box>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        ))}
      </Box>
    </Container>
  );
};

export default Post;


async function handleLike(postId) {
  try {
    const token = localStorage.getItem('token');
    const response = await axios.post(`/api/posts/${postId}/like`, {}, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    fetchPosts(); // Refresh posts to reflect the new like count
  } catch (err) {
    console.error('Error liking post:', err);
  }
}

async function handleComment(postId) {
  try {
    const token = localStorage.getItem('token');
    await axios.post(`/api/posts/${postId}/comment`, { text: commentText }, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    setCommentText(''); // Clear comment input
    fetchPosts(); // Refresh posts to reflect the new comment
  } catch (err) {
    console.error('Error commenting on post:', err);
  }
}