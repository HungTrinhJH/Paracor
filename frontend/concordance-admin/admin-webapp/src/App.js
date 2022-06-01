import React, { Component, Fragment } from "react";
import "./App.css";
import Homepage from "./Screens/Homepage/Homepage";
import { connect } from "react-redux";
import Login from "./Screens/Login";
import { BrowserRouter, Route, Switch, Redirect } from "react-router-dom";
import { createAciton } from "./Redux/Action";
import { FETCH_CREDENTIALS } from "./Redux/Action/type";
import ProtectedRoute from "./ProtectedRoute";
class App extends Component {
  // Len local storegare lay thong tin user neu da dang nhap
  _getCridentialsFromLocal = () => {
    const credentialString = localStorage.getItem("credentials");
    if (credentialString) {
      this.props.dispatch(
        createAciton(FETCH_CREDENTIALS, JSON.parse(credentialString))
      );
    }
  };

  componentDidMount() {
    this._getCridentialsFromLocal();
    if (!localStorage) {
      return <Redirect to="/login" />;
    }
  }
  render() {
    return (
      <BrowserRouter>
        <Switch>
          <Route path="/login">
            <Login />
          </Route>
          <ProtectedRoute path="/home">
            <Homepage />
          </ProtectedRoute>
          <Route exact path="/">
            <Redirect exact from="/" to="home" />
          </Route>
          <Route path="*">
            <Redirect from="/" to="home" />
          </Route>
        </Switch>
      </BrowserRouter>
    );
  }
}

const mapStateToProps = (state) => {
  return {
    isAuthenticated: state.user.isAuthenticated,
  };
};

export default connect(mapStateToProps)(App);
