import React, { Component } from "react";
import { connect } from "react-redux";
import { createAction } from "../../../../Redux/Action";
import {
  RESET_LOADING,
  SEARCH_TYPE,
  SET_ALERT,
} from "../../../../Redux/Action/type";
import { dataService } from "../../../../Services";
import "./SearchController.css";
import Input from "../../../../Components/Form/Input";
import RadioButton from "../../../../Components/Form/RadioButton";
import Checkbox from "../../../../Components/Form/Checkbox";
import Dropdown from "../../../../Components/Form/Dropdown";
class SearchController extends Component {
  state = {
    searchValue: "",
    searchType: "mat",
    loaded: false,
    posValue: "",
    nerValue: "",
    posAcceptance: false,
    nerAcceptance: false,
  };

  componentDidUpdate(prevProps) {
    if (
      this.props.currentPage !== prevProps.currentPage &&
      this.props.currentPage !== 1
    ) {
      this.updateSearch(this.props.currentPage);
    }
  }

  //Handle onChange event
  handleChange = (key) => (value) => {
    if (key === "searchType") {
      switch (value) {
        case "mat":
          this.props.handleSearchType("Match");
          break;
        case "mor":
          this.props.handleSearchType("Morphological");
          break;
        case "phrase":
          this.props.handleSearchType("Phrase");
          break;
        default:
          break;
      }
    }
    if ((key === "posAcceptance" || key === "nerAcceptance") && !value) {
      if (key === "posAcceptance") {
        this.setState({
          [key]: value,
          posValue: "",
        });
      } else {
        this.setState({
          [key]: value,
          nerValue: "",
        });
      }
    } else {
      this.setState({ [key]: value });
    }
  };

  updateSearch = (page) => {
    let lang = this.props.language === "vietnamese" ? "vn" : "en";
    let tag = { pos: this.state.posValue, ner: this.state.nerValue };
    this.props.dispatch(
      createAction(RESET_LOADING, {
        loaded: true,
        msg: "Search by word is loading...",
      })
    );
    let searchType = "";
    let { nerValue, posValue } = this.state;
    if (this.state.searchType === "mat") {
      searchType = "Match " + posValue + " " + nerValue;
    } else if (this.state.searchType === "mor") {
      searchType = "Morphological " + posValue + " " + nerValue;
    } else {
      searchType = "Phrase " + posValue + " " + nerValue;
    }

    dataService
      .fetchData_Search(
        this.state.searchValue,
        this.state.searchType,
        lang,
        tag,
        page
      )
      .then((res) => {
        let data = {};
        if (lang === "vietnamese") {
          data.source = res.data.target;
          data.target = res.data.source;
        } else data = res.data;
        this.props.dispatch(createAction(RESET_LOADING, false));
        this.props.dispatch(createAction(SEARCH_TYPE, searchType));
        this.props.getSearchData(data);
      })
      .catch((err) => {
        this.props.dispatch(createAction(RESET_LOADING, false));
        throw new Error(err);
      });
  };

  // Submit search:
  // Request to server,
  // then get data if
  //    1 => dispatch data to Redux ,
  //    0 => Modal : alert to client
  handleOnsubmit = (e) => {
    e.preventDefault();
    this.props.handleRowSelected(null);
    if (!this.state.searchValue) {
      const msg = "Please enter the keyword before searching";
      this.props.dispatch({ type: SET_ALERT, payload: msg });
      setTimeout(() => {
        this.props.dispatch({ type: SET_ALERT, payload: null });
      }, 3000);
    } else {
      this.props.resetCurrentPage(1);
      this.updateSearch(1);
    }
  };

  // Get tag value base on tag and language
  getTagsValue = (typeTag) => {
    let type = this.props.language.charAt(0) + typeTag;
    return this.props.tags[`${type}`].map((item, index) => {
      let desc = item.description;
      if (item.description.length >= 30) {
        desc = item.description.slice(0, 30) + "...";
      }
      return { value: item.tag, label: desc };
    });
  };

  // Set State when Refresh button is trigged
  handleRefresh = () => {
    if (this.state.loaded) {
      const msg = "Data is loading from server, please wait before refreshing";
      this.props.dispatch({ type: SET_ALERT, payload: msg });
      setTimeout(() => {
        this.props.dispatch({ type: SET_ALERT, payload: null });
      }, 3000);
    } else {
      this.props.refreshHandler();
      this.setState({
        searchValue: "",
        searchType: "mat",
        loaded: false,
        posValue: "",
        nerValue: "",
        posAcceptance: false,
        nerAcceptance: false,
      });
    }
  };

  render() {
    //State
    const {
      searchValue,
      searchType,
      posAcceptance,
      nerAcceptance,
      posValue,
      nerValue,
    } = this.state;

    //Get data for Dropdown component: eNer, vNer, ePos or vPos
    let nerData = nerAcceptance ? this.getTagsValue("Ner") : [];
    let posData = posAcceptance ? this.getTagsValue("Pos") : [];

    return (
      <div className="col-10 seach__controller mt-3">
        {/* Form Search */}
        <form className="row" onSubmit={this.handleOnsubmit}>
          {/*  Word Controller*/}
          <div className="col-5">
            <p className="content__title">Word</p>
            <Input
              type="text"
              placeholder="Enter keyword to search..."
              onChange={this.handleChange("searchValue")}
              value={searchValue}
            />
            <div className="tag__choosen d-flex justify-content- algin-items-center">
              <RadioButton
                label="Match"
                onChange={this.handleChange("searchType")}
                selected={searchType === "mat"}
                value="mat"
                styleClass="mr-3"
              />
              <RadioButton
                label="Morphological"
                onChange={this.handleChange("searchType")}
                selected={searchType === "mor"}
                value="mor"
                styleClass="mr-3"
              />
              <RadioButton
                label="Phrase"
                onChange={this.handleChange("searchType")}
                selected={searchType === "phrase"}
                value="phrase"
                styleClass="mr-3"
              />
            </div>
          </div>
          {/* End Word Controller*/}

          {/* Tag  Controller*/}
          <div className="search__tag col-4 px-auto">
            <p className="content__title">Tag</p>
            <div>
              <div className="pos d-flex align-items-center mb-3">
                <Checkbox
                  label="Pos"
                  selected={posAcceptance}
                  onChange={this.handleChange("posAcceptance")}
                  styleClass="mr-3"
                />
                <Dropdown
                  onChange={this.handleChange("posValue")}
                  placeholder="Select pos"
                  data={posData}
                  value={posValue}
                />
              </div>
              <div className="ner d-flex align-items-center">
                <Checkbox
                  label="Ner"
                  selected={nerAcceptance}
                  onChange={this.handleChange("nerAcceptance")}
                  styleClass="mr-3"
                />
                <Dropdown
                  onChange={this.handleChange("nerValue")}
                  placeholder="Select ner"
                  data={nerData}
                  value={nerValue}
                />
              </div>
            </div>
          </div>
          {/* End Tag controller */}

          {/* Button: submit  & refresh */}
          <div className="col-3 btn-search-wrp ml-auto">
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
          {/* End btn submit */}
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

// Connect component SearchController with Redux store
export default connect(mapStateToProps)(SearchController);
