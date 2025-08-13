# Contributing to the Blue Water Macro Capstone Simulator

We welcome contributions from the quantitative finance community! This framework is designed to advance education in quantitative trading and we encourage improvements that benefit students and researchers worldwide.

## 🎯 Contribution Focus

**Educational Enhancements Preferred**: We especially welcome contributions that improve the educational value of the framework, such as:
- Better explanations and docstrings
- Additional tutorial content
- New educational examples
- Improved error messages and debugging aids
- Enhanced documentation

## 🚀 Getting Started

### Quick Start for Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/capstone_simulator.git
   cd capstone_simulator
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies** including development tools:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov  # For testing
   ```
5. **Run tests** to ensure everything works:
   ```bash
   pytest tests/
   ```

## 📝 How to Contribute

### 🐛 Reporting Bugs

**Submit Issues for Bugs**: If you find a bug, please create a GitHub issue with:
- **Clear title**: Brief description of the problem
- **Steps to reproduce**: Exact steps that trigger the bug
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment details**: Python version, OS, package versions
- **Minimal example**: Small code snippet that demonstrates the issue

**Template for Bug Reports**:
```markdown
## Bug Description
Brief description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version:
- OS:
- Package versions: (run `pip freeze`)

## Minimal Example
```python
# Your minimal code example here
```

### 💡 Feature Requests

**Submit PRs for Features**: We welcome feature requests, especially those that enhance educational value:

- **New educational content**: Tutorials, examples, explanations
- **Additional metrics**: New performance measures or risk metrics
- **Enhanced visualizations**: Better plots, charts, or reports
- **Data source integrations**: Support for additional data providers
- **Model implementations**: New ML models or strategies
- **Testing improvements**: Additional test coverage or test utilities

### 🔧 Code Contributions

#### Before You Start
- **Check existing issues** to avoid duplicate work
- **Discuss major changes** by creating an issue first
- **Focus on educational value** - how does this help students learn?

#### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards:
   - Add comprehensive docstrings with educational context
   - Include type hints where appropriate
   - Follow existing code style and patterns
   - Add unit tests for new functionality

3. **Write tests**:
   ```bash
   # Add tests in the appropriate test file
   tests/test_your_module.py
   
   # Run tests to ensure they pass
   pytest tests/test_your_module.py -v
   ```

4. **Update documentation**:
   - Update README.md if needed
   - Add docstrings explaining the educational purpose
   - Include examples in docstrings

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: brief description of your change"
   ```

6. **Push and create Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## 📋 Coding Standards

### Code Style
- **Follow PEP 8** for Python code style
- **Use meaningful variable names** that are self-documenting
- **Add type hints** for function parameters and return values
- **Keep functions focused** - one responsibility per function

### Documentation Standards
- **Educational docstrings**: Explain not just what, but why and how it relates to quantitative finance
- **Include examples**: Show practical usage in docstrings
- **Educational notes**: Add "Educational Note:" sections explaining financial concepts
- **Clear error messages**: Help users understand what went wrong and how to fix it

### Example of Good Docstring:
```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio for a return series.
    
    The Sharpe ratio measures risk-adjusted performance by dividing excess returns
    by volatility. It's one of the most important metrics in quantitative finance.
    
    Educational Note:
        A Sharpe ratio > 1.0 is generally considered good, > 2.0 is excellent.
        The ratio helps compare strategies with different risk profiles.
    
    Args:
        returns: Daily return series
        risk_free_rate: Annual risk-free rate (default: 0.0)
        
    Returns:
        Annualized Sharpe ratio
        
    Example:
        >>> returns = pd.Series([0.01, -0.005, 0.02, -0.01])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe ratio: {sharpe:.2f}")
    """
```

### Testing Standards
- **Write unit tests** for all new functions
- **Include edge cases** (empty data, NaN values, extreme values)
- **Test error handling** - ensure functions fail gracefully
- **Use descriptive test names** that explain what is being tested

## 🎓 Educational Contribution Guidelines

### What We're Looking For
- **Clear explanations**: Help students understand complex concepts
- **Practical examples**: Real-world applications of quantitative methods
- **Interactive content**: Jupyter notebooks with step-by-step walkthroughs
- **Best practices**: Industry-standard approaches to common problems
- **Career guidance**: Insights into quantitative finance careers

### Educational Standards
- **Assume basic knowledge**: Target upper-level undergraduate/graduate students
- **Explain financial concepts**: Don't assume deep finance knowledge
- **Provide context**: Explain why techniques are used in practice
- **Include warnings**: Point out common pitfalls and mistakes
- **Connect to industry**: Relate concepts to real-world applications

## 🧪 Testing Your Contributions

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Run integration tests only

# Run tests for specific modules
pytest tests/test_utils_simulate.py -v
```

### Test Categories
- **Unit tests**: Test individual functions
- **Integration tests**: Test component interactions
- **Performance tests**: Ensure reasonable execution times
- **Educational tests**: Verify examples work correctly

## 📊 Performance Considerations

- **Consider computational efficiency** for educational use
- **Profile performance** for functions processing large datasets
- **Add performance tests** for time-critical functions
- **Document complexity** for algorithms with non-linear scaling

## 🌟 Recognition

Contributors who make significant educational improvements may be:
- **Acknowledged in documentation** and release notes
- **Invited to collaborate** on educational content
- **Considered for opportunities** at Blue Water Macro Corp.

## 🤝 Community Guidelines

### Be Respectful
- **Constructive feedback**: Focus on code and ideas, not people
- **Inclusive language**: Welcome contributors of all backgrounds
- **Patient teaching**: Help newcomers learn and improve

### Be Professional
- **Educational focus**: Keep discussions relevant to quantitative education
- **Evidence-based**: Support suggestions with reasoning or references
- **Collaborative spirit**: Work together to improve the framework

## 📞 Getting Help

### For Contributors
- **GitHub Discussions**: Ask questions and discuss ideas
- **Code Reviews**: Request feedback on your contributions
- **Educational Support**: Get help with quantitative finance concepts

### Contact Information
- **Technical questions**: Create a GitHub issue
- **Educational collaboration**: education@bluewatermacro.com
- **General inquiries**: contact@bluewatermacro.com

## 📄 License

By contributing, you agree that your contributions will be licensed under the same Blue Water Macro Educational License (BWMEL) as the project. See [LICENSE](LICENSE) for details.

### Key Points:
- **Educational use**: Contributions support educational mission
- **Attribution required**: Contributors will be acknowledged
- **Commercial licensing**: Separate terms for commercial use

## 🎯 Roadmap and Priorities

### Current Priorities
1. **Enhanced educational content** and tutorials
2. **Additional performance metrics** and risk measures
3. **Better visualization tools** for strategy analysis
4. **More comprehensive testing** and error handling
5. **Integration with additional data sources**

### Future Directions
- **Advanced portfolio construction** techniques
- **Risk management** modules and tools
- **Alternative data integration** capabilities
- **Real-time simulation** features
- **Interactive web interface** for beginners

---

## Thank You!

Thank you for contributing to quantitative finance education! Your efforts help students worldwide develop the skills needed for successful careers in professional asset management and quantitative research.

**Together, we're building the next generation of quantitative finance professionals.**

*Blue Water Macro Corp. - Advancing Quantitative Finance Education*