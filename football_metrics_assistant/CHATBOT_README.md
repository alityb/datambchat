# Football Metrics Assistant Chatbot

A modern, responsive chatbot frontend for the Football Metrics Assistant that provides an intuitive way to interact with football analytics data.

## Features

- 🎨 **Modern UI Design** - Clean, professional interface matching the existing DataMB style
- 🌙 **Dark/Light Mode** - Toggle between themes with persistent preference storage
- 💬 **Real-time Chat** - Interactive chat interface with typing indicators
- 📱 **Responsive Design** - Works seamlessly on desktop and mobile devices
- ⚡ **Fast API Integration** - Direct integration with the existing football metrics API
- 📊 **Smart Data Display** - Automatic table formatting for structured data
- 🎯 **Example Queries** - Clickable example questions to get started quickly

## Quick Start

### 1. Start the Chatbot Server

```bash
cd football_metrics_assistant
python3 chatbot_server.py
```

### 2. Access the Chatbot

- **Chatbot Frontend**: http://localhost:8080
- **API Endpoints**: http://localhost:8080/api
- **Health Check**: http://localhost:8080/health

### 3. Start Chatting

The chatbot will load with example queries you can click on, or type your own questions about:
- Player statistics and performance
- Team comparisons and analysis
- League standings and metrics
- Statistical definitions and explanations

## Example Queries

- "Show me the top 10 players by goals scored"
- "Compare teams in the Premier League"
- "What are the best defenders in La Liga?"
- "Explain what xG means in football"
- "Show players with more than 20 assists"

## Technical Details

### Frontend Technologies
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with CSS variables and animations
- **JavaScript (ES6+)** - Async/await, modern DOM manipulation
- **Font Awesome** - Icons for better UX
- **Inter Font** - Professional typography

### Backend Integration
- **FastAPI** - High-performance Python web framework
- **CORS Support** - Cross-origin resource sharing enabled
- **API Mounting** - Seamless integration with existing football metrics API
- **Static File Serving** - Efficient delivery of frontend assets

### Key Features Implementation

#### Dark Mode Toggle
- CSS variables for theme switching
- Local storage persistence
- Smooth transitions between themes
- Automatic icon changes (moon/sun)

#### Chat Interface
- Real-time message display
- Typing indicators with animated dots
- Auto-resizing text input
- Message timestamps
- Avatar system for user/bot distinction

#### Data Handling
- Automatic table detection and formatting
- Responsive table design
- Error handling and user feedback
- Loading states and indicators

## File Structure

```
football_metrics_assistant/
├── chatbot.html          # Main chatbot frontend
├── chatbot_server.py     # Server to serve frontend + API
├── main.py              # Existing football metrics API
├── llm_interface.py     # Gemini Flash integration
├── tools.py             # Data analysis tools
├── preprocessor.py      # Query preprocessing
└── CHATBOT_README.md    # This file
```

## Customization

### Styling
The chatbot uses CSS variables for easy customization. Key variables include:
- `--primary-color`: Main brand color
- `--secondary-color`: Accent color
- `--background-color`: Page background
- `--container-bg`: Container backgrounds
- `--text-color`: Text color
- `--border-color`: Border colors

### Adding New Features
1. **New Message Types**: Extend the `addMessage()` function
2. **Additional Styling**: Add new CSS classes and variables
3. **API Integration**: Modify the `sendMessage()` function
4. **UI Components**: Create new HTML elements and corresponding styles

## Browser Support

- **Chrome**: 80+
- **Firefox**: 75+
- **Safari**: 13+
- **Edge**: 80+

## Performance

- **Lightweight**: Minimal dependencies, fast loading
- **Responsive**: Optimized for various screen sizes
- **Efficient**: CSS animations use GPU acceleration
- **Accessible**: Proper ARIA labels and keyboard navigation

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port in chatbot_server.py
   port=8081  # or any available port
   ```

2. **API Connection Errors**
   - Ensure the main football metrics API is working
   - Check that the API is mounted at `/api`
   - Verify CORS settings

3. **Styling Issues**
   - Clear browser cache
   - Check CSS variable definitions
   - Verify Font Awesome CDN connection

### Debug Mode
Enable debug logging by modifying the uvicorn run call:
```python
uvicorn.run(
    "chatbot_server:app",
    host="0.0.0.0",
    port=8080,
    reload=True,
    log_level="debug"  # Change to debug
)
```

## Contributing

To contribute to the chatbot:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

## License

This chatbot frontend is part of the Football Metrics Assistant project and follows the same licensing terms.

---

**Enjoy chatting with your AI Football Analytics Expert! ⚽🤖** 