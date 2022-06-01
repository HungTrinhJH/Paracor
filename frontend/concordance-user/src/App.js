import React, { Component, Suspense } from "react";
import "./App.css";
import { BrowserRouter, Route, Switch } from "react-router-dom";
import Header from "./Layouts/Header";
import { connect } from "react-redux";
import { createAction } from "./Redux/Action";
import {
  FETCH_EN_DATA,
  FETCH_VI_DATA,
  RESET_LOADING,
  UPDATE_TOTAL_PAGE_HOME,
} from "./Redux/Action/type";
import { dataService } from "./Services";
import Footer from "./Layouts/Footer";
import Spinner from "./Components/Spinner";
import ErrorBoundary from "./Components/ErrorBoundary";
import LazyPreload from "./LazyPreload";

const NotFound = LazyPreload(() => import("./Components/NotFound"));
const User = LazyPreload(() => import("./Screens/User"));
const Corpus = LazyPreload(() => import("./Screens/Corpus"));
const Home = LazyPreload(() => import("./Screens/Home"));
const FAQ = LazyPreload(() => import("./Screens/FAQ"));

// Routes
const routes = [
  { path: "/", exact: true, component: Home },
  { path: "/user", exact: true, component: User },
  { path: "/corpus", exact: false, component: Corpus },
  { path: "/faq", exact: true, component: FAQ },
  { path: "*", exact: false, component: NotFound },
];

class App extends Component {
  componentDidMount() {
    this.fetchDataSentence();
  }
  componentDidUpdate(prevProps) {
    if (this.props.pageNumber !== prevProps.pageNumber) {
      this.fetchDataSentence();
    }
  }
  fetchDataSentence = () => {
    let page = this.props.pageNumber;
    this.props.dispatch(
      createAction(RESET_LOADING, {
        loaded: true,
        msg: "Data is loading...",
      })
    );
    dataService
      .fetchDataHome(page)
      .then((res) => {
        if (res.length === 2) {
          this.props.dispatch(createAction(FETCH_EN_DATA, res[0].data.results));
          this.props.dispatch(createAction(FETCH_VI_DATA, res[1].data.results));
          this.props.dispatch(
            createAction(UPDATE_TOTAL_PAGE_HOME, res[0].data.count)
          );  
          this.props.dispatch(
            createAction(RESET_LOADING, {
              state: false,
              msg: null,
            })
          );
        }
      })
      .catch((err) => {
        this.props.dispatch(
          createAction(RESET_LOADING, { state: false, msg: null })
        );
        throw new Error(err);
      });
  };

  render() {
    return (
      <BrowserRouter className="wrapper">
        <Header />
        <Suspense fallback={<div>Loading...</div>}>
          <Switch>
            {routes.map((route) => (
              <Route
                key={route.path}
                exact={route.exact}
                path={route.path}
                render={() => (
                  <ErrorBoundary>
                    <route.component></route.component>
                  </ErrorBoundary>
                )}
              />
            ))}
          </Switch>
          {this.props.loaded === true ? <Spinner /> : null}
        </Suspense>
        {/* <Footer /> */}
      </BrowserRouter>
    );
  }
}
const mapStateToProps = (state) => {
  return {
    pageNumber: state.Controller.currentPageHome,
    loaded: state.Controller.loaded,
  };
};
export default connect(mapStateToProps)(App);
