import React, { Component } from "react";
import "./ErrorBoundary.css";
class ErrorBoundary extends Component {
  state = { hasError: false };

  componentDidCatch(error, info) {
    // Display fallback UI
    console.log(error);
    this.setState({ hasError: true });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div class="page_500_wrp">
          <div className="page_500_container">
            <img src="https://i.imgur.com/qIufhof.png" alt="error" />
            <h1>
              <span>500</span> <br />
              Internal server error
            </h1>
            <p>We are currently trying to fix the problem.</p>
            <p className="info">Sorry for this inconvenient</p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
export default ErrorBoundary;
