import React, { Component } from "react";
import { connect } from "react-redux";
import { Badge } from "../../../../Components/Badge";
import Dropdown from "../../../../Components/Form/Dropdown";
import { createAction } from "../../../../Redux/Action";
import { RESET_LOADING, SET_ALERT } from "../../../../Redux/Action/type";
import { dataService } from "../../../../Services";

import "./SearchStructureController.css";
class SearchStructureController extends Component {
  state = {
    posValue: "",
    searchList: [],
    selectedFile: null,
    errors: false,
  };

  // Handle Change
  handleChange = (key) => (value) => {
    this.setState({
      [key]: value,
      searchList: [...this.state.searchList, value],
    });
  };

  //Handle Chane file
  handleChangeFile = (event) => {
    this.props.dispatch(
      createAction(RESET_LOADING, {
        loaded: false,
        msg: null,
      })
    );
    this.props.dispatch({
      type: SET_ALERT,
      payload: "This feature is developing, sorry for this inconvenient",
    });
    setTimeout(() => {
      this.props.dispatch({ type: SET_ALERT, payload: null });
    }, 3000);

    // let fileName = event.target.files[0].name;
    // let isTxtFile = fileName.includes(".txt");
    // this.setState({
    //   selectedFile: event.target.files[0],
    //   errors: !isTxtFile,
    // });
  };

  // Delete badge
  deleteHandler = (title) => {
    let newSearchList = this.state.searchList.filter((item) => item !== title);
    this.setState({
      searchList: [...newSearchList],
    });
  };

  // handleRefresh
  handleRefresh = () => {
    this.props.refreshHandler();
    this.setState({
      searchList: [],
      selectedFile: null,
      errors: false,
      errorsMsg: "",
      posValue: "",
    });
  };

  // Get tag value base on tag and language
  getTagsValue = (typeTag) => {
    let type = this.props.language.charAt(0) + typeTag;
    let tagValues = this.props.tags[`${type}`].map((item, index) => {
      return { value: item.tag, label: item.description };
    });
    return tagValues;
  };
  componentDidUpdate(prevProps) {
    if (
      this.props.currentPage !== prevProps.currentPage &&
      this.props.currentPage !== 1
    ) {
      this.fetchSearchStructure(this.props.currentPage);
    }
  }
  fetchSearchStructure = (page) => {
    let lang = this.props.language === "vietnamese" ? "vn" : "en";
    this.props.dispatch(
      createAction(RESET_LOADING, {
        loaded: true,
        msg: "Search by structure is loading...",
      })
    );

    // Muti structure searching
    if (this.state.selectedFile !== null) {
      // const fd = new FormData();
      // fd.append("filename", this.state.selectedFile);
      // dataService
      //   .fetchSearchMultiStructure(fd)
      //   .then((res) => {
      //     this.props.getAllData(res.data);
      //     this.props.dispatch(createAction(RESET_LOADING, false));
      //   })
      //   .catch((err) => {
      //     this.props.dispatch(createAction(RESET_LOADING, false));
      //     throw new Error(err);
      //   });
    } else
      dataService
        .fetchSearchStructure(lang, this.state.searchList, page)
        .then((res) => {
          this.props.getAllData(res.data);
          this.props.dispatch(
            createAction(RESET_LOADING, {
              loaded: false,
              msg: null,
            })
          );
        })
        .catch((err) => {
          this.props.dispatch(
            createAction(RESET_LOADING, {
              loaded: false,
              msg: null,
            })
          );
          throw new Error(err);
        });
  };
  // OnSubmit
  handleOnsubmit = (e) => {
    e.preventDefault();

    const { searchList, selectedFile, errors } = this.state;
    if (searchList.length === 0 && selectedFile === null) {
      const msg = "Please choose the POS, or import file before searching";
      this.props.dispatch({ type: SET_ALERT, payload: msg });
      setTimeout(() => {
        this.props.dispatch({ type: SET_ALERT, payload: null });
      }, 3000);
    } else if (selectedFile !== null && errors) {
      const msg =
        "Please change the valid file, the system just receive .txt file!";
      this.props.dispatch({ type: SET_ALERT, payload: msg });
      setTimeout(() => {
        this.props.dispatch({ type: SET_ALERT, payload: null });
      }, 3000);
    } else {
      if (!this.state.errors) {
        this.props.resetCurrentPage(1);
        this.fetchSearchStructure(1);
      }
    }
  };

  render() {
    const { posValue } = this.state;

    //Get data for Dropdown component: eNer, vNer, ePos or vPos
    let posData = this.getTagsValue("Pos");
    const searchList = this.state.searchList.map((tag, index) => {
      return (
        <Badge title={tag} key={index} deleteHandler={this.deleteHandler} />
      );
    });

    return (
      <div className="col-10 seach__controller mt-3">
        {/* Form Search by phrase */}
        <form className="row" onSubmit={this.handleOnsubmit}>
          {/* Tag  Controller*/}
          <div className="search__tag col-3 px-auto">
            <p className="content__title">Single structue</p>
            <div>
              <div className="pos d-flex align-items-center">
                <Dropdown
                  onChange={this.handleChange("posValue")}
                  placeholder="Select pos"
                  data={posData}
                  value={posValue}
                  disabled={this.state.selectedFile !== null ? true : false}
                />
              </div>
            </div>
          </div>
          {/* End Tag controller */}

          {/* List search */}
          <div className="col-4 search_phrase bg-light d-flex p-2 flex-wrap  border">
            {searchList}
          </div>
          <div className="col-3">
            <p className="content__title">
              Mutil structue
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
                disabled={this.state.searchList.length > 0 ? true : false}
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
          <div className="col-2 btn-search-wrp">
            {/* Submit data and post to server */}
            <div className="my-1">
              <button type="submit" className="btn-search">
                SEARCH
              </button>
            </div>

            {/* Refresh state to initState */}
            <div className="my-1">
              <button
                type="button"
                className="btn-refresh"
                onClick={this.handleRefresh}
                disabled={this.props.loaded}
              >
                REFRESH
              </button>
            </div>
          </div>
        </form>
        {/* End form search */}
      </div>
    );
  }
}

// Get State from Redux store to props
const mapStateToProps = (state) => {
  return {
    tags: state.Tag.tags,
    loaded: state.Controller.loaded,
  };
};

export default connect(mapStateToProps)(SearchStructureController);
