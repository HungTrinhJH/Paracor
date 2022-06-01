import React, { Component } from "react";
import Input from "../../../Components/Form/Input";
import { connect } from "react-redux";
import { createAction } from "../../../Redux/Action";
import { SET_ALERT } from "../../../Redux/Action/type";
class UserController extends Component {
  state = {
    search: {
      source: "",
      target: "",
    },
    selectedFile: null,
    errors: false,
  };

  //Handle Chane file
  handleChangeFile = (event) => {
    let fileName = event.target.files[0].name;
    let isTxtFile = fileName.includes(".txt");
    this.setState({
      selectedFile: event.target.files[0],
      errors: !isTxtFile,
    });
  };

  handleChange = (key) => (value) => {
    this.setState({
      ...this.state,
      search: {
        ...this.state.search,
        [key]: value,
      },
    });
  };
  handleSubmit = (e) => {
    e.preventDefault();
    let { selectedFile, search } = this.state;
    let isError =
      (search.source === "" || search.target === "") && selectedFile === null;
    if (isError) {
      const msg =
        "Please fill the source and target sentence or select file before importing data";
      this.props.dispatch(createAction(SET_ALERT, msg));
      setTimeout(() => {
        this.props.dispatch({ type: SET_ALERT, payload: null });
      }, 3000);
    }
  };
  handleRefresh = () => {
    this.setState({
      search: {
        source: "",
        target: "",
      },
      selectedFile: null,
      errors: false,
    });
  };
  render() {
    let { search } = this.state;
    return (
      <div className="mx-5">
        <form onSubmit={this.handleSubmit}>
          <div className="row">
            <div className="col-7">
              <p className="content__title">Source sentence</p>
              <Input
                type="text"
                placeholder="Enter the source sentence..."
                onChange={this.handleChange("source")}
                value={search.source}
                disabled={this.state.selectedFile !== null ? true : false}
              />
              <p className="content__title">Target sentence</p>

              <Input
                type="text"
                placeholder="Enter the target sentence..."
                onChange={this.handleChange("target")}
                value={search.target}
                disabled={this.state.selectedFile !== null ? true : false}
              />
            </div>

            <div className="col-5 d-flex flex-column justify-content-between">
              <div>
                <p className="content__title">
                  Import file
                  <a
                    href="/assets/multi_structure_example.txt"
                    download
                    className="text-secondary"
                  >
                    <i
                      className="fa fa-question-circle ml-2"
                      data-toggle="tooltip"
                      data-placement="right"
                      title="Click here to download the file input format"
                      data-animation="true"
                      data-delay="100"
                    ></i>
                  </a>
                </p>
                <div className="custom-file mb-2">
                  <input
                    type="file"
                    className="custom-file-input"
                    onChange={this.handleChangeFile}
                    name="selectedFile"
                    disabled={
                      this.state.search.source.length > 0 ||
                      this.state.search.target.length > 0
                        ? true
                        : false
                    }
                  />
                  <label className="custom-file-label">
                    {this.state.selectedFile == null
                      ? "Choosen file"
                      : this.state.selectedFile.name}
                  </label>
                  {this.state.errors && (
                    <span className="text-danger">File is not valid!</span>
                  )}
                </div>
              </div>
              <div className="d-flex" style={{ marginBottom: "16px" }}>
                <button type="submit" className="btn-search mr-2">
                  IMPORT
                </button>
                <button
                  type="button"
                  className="btn-refresh mr-2"
                  onClick={this.handleRefresh}
                  disabled={this.props.loaded}
                >
                  REFRESH
                </button>
                <button
                  type="button"
                  className="btn-refresh"
                  disabled={this.props.loaded}
                  data-toggle="tooltip"
                  data-placement="right"
                  title="Download corpus is aligned"
                >
                  DOWNLOAD
                </button>
              </div>
            </div>
          </div>
        </form>
      </div>
    );
  }
}

export default connect()(UserController);
