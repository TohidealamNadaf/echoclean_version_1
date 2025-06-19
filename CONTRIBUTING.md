# Contributing to EchoClean

Thank you for your interest in contributing to EchoClean! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/echoclean.git
   cd echoclean
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask flask-sqlalchemy pydub scikit-learn
   pip install pytest pytest-cov  # For testing
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## Code Style Guidelines

### Python Code Style
- Follow PEP 8 conventions
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use descriptive variable and function names

### JavaScript Code Style
- Use ES6+ features
- Consistent indentation (2 spaces)
- Use async/await for asynchronous operations
- Add JSDoc comments for complex functions

### CSS Style
- Use Bootstrap utilities first
- Custom CSS only when necessary
- Follow BEM naming convention for custom classes
- Ensure dark theme compatibility

## Audio Processing Guidelines

### Feature Extraction
- Maintain 29-dimensional feature vector structure
- Document any new features thoroughly
- Ensure real-time performance (<500ms processing)
- Add proper error handling for edge cases

### Similarity Algorithms
- Preserve multi-metric approach (cosine, euclidean, correlation)
- Test accuracy improvements with sample datasets
- Document threshold changes with reasoning
- Maintain backward compatibility

## Testing Requirements

### Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=. tests/
```

### Audio Testing
- Test with various audio formats (WAV, MP3, FLAC, M4A, OGG)
- Verify sample rates from 8kHz to 48kHz
- Test with different audio lengths (0.5s to 60s)
- Include edge cases (silence, noise, very short clips)

### Integration Tests
- Test complete upload → analysis → result workflow
- Verify real-time recording functionality
- Test session management and cleanup
- Validate API response formats

## Submission Guidelines

### Pull Request Process
1. Create a feature branch from main
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Update documentation if needed
5. Ensure all tests pass
6. Submit pull request with detailed description

### Commit Message Format
```
type(scope): brief description

Detailed explanation of changes if needed

Fixes #issue_number
```

Types: feat, fix, docs, style, refactor, test, chore

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed

## Areas for Contribution

### High Priority
- Additional audio format support
- Performance optimizations
- Enhanced deepfake detection algorithms
- Mobile browser compatibility
- Accessibility improvements

### Medium Priority
- Batch processing capabilities
- Audio quality metrics
- Advanced visualization features
- Export/import functionality
- Multi-language support

### Low Priority
- Alternative UI themes
- Additional statistical metrics
- Integration examples
- Documentation improvements

## Bug Reports

When reporting bugs, include:
- Operating system and version
- Python version
- Browser (for frontend issues)
- Steps to reproduce
- Expected vs actual behavior
- Audio file details (if relevant)
- Error messages and logs

## Feature Requests

For new features, provide:
- Use case description
- Expected behavior
- Implementation suggestions
- Potential impact on existing features
- Performance considerations

## Security Issues

Report security vulnerabilities privately to:
- Email: security@echoclean.example.com
- Do not create public issues for security problems
- Allow reasonable time for response before disclosure

## Documentation Contributions

- Update README.md for major changes
- Add inline code comments for complex algorithms
- Create examples for new features
- Improve API documentation
- Update configuration guides

## Performance Guidelines

### Audio Processing
- Target <500ms for standard analysis
- <200ms for fast mode processing
- Memory usage <100MB per concurrent user
- Support for 10+ concurrent sessions

### Frontend Performance
- Initial page load <2 seconds
- Audio upload response <1 second
- Real-time analysis feedback <500ms
- Mobile device compatibility

## License

By contributing to EchoClean, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Open an issue for general questions
- Join discussions in existing issues
- Contact maintainers for guidance
- Check existing documentation first

Thank you for contributing to EchoClean!