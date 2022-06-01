import React, { Component } from "react";
import { connect } from "react-redux";
import "./StatisController.css";
import { dataService } from "../../../Services";
import { createAction } from "../../../Redux/Action";
import RadioButton from "../../../Components/Form/RadioButton";
import { RESET_LOADING } from "../../../Redux/Action/type";
class StatisController extends Component {
  state = {
    num: "top", // 1. all , 2. top
    lang: "en", // en/vn
    count: 100, // count if top words is top
    typeTag: "", // Ner or Pos
    typeTagDetail: "", // Detail Ner or Pos
  };

  handleChange = (key) => (value) => {
    if (key === "lang") {
      this.props.setLang(value);
    }
    this.setState({
      [key]: value,
    });
  };

  onChangleHandler = (e) => {
    this.setState({
      [e.target.name]: e.target.value,
    });
  };

  refreshHandler = () => {
    this.setState({
      num: "top",
      lang: "en",
      count: 100,
      typeTag: "",
      typeTagDetail: "",
    });
  };

  onSubmitHandler = (e) => {
    let { num, lang, count, typeTag, typeTagDetail } = this.state;
    e.preventDefault();
    // When submit event is trigged, client will be call API to server
    this.props.dispatch(
      createAction(RESET_LOADING, {
        loaded: true,
        msg: "Statistical data is loading...",
      })
    );
    dataService
      .fetchData_QueryStatistic(num, lang, count, typeTag, typeTagDetail)
      .then((response) => {
        this.props.queryData(response.data);
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
  render() {
    const { num, lang, typeTag, typeTagDetail } = this.state;
    let options = null;
    // Get typeTag : pos, ner
    let typeTagHandle =
      this.state.lang.charAt(0) +
      this.state.typeTag.charAt(0).toUpperCase() +
      this.state.typeTag.slice(1, this.state.typeTag.length);
    let tagDetail = null;
    // Get typeTagDetail
    if (this.state.typeTag !== "") {
      tagDetail = this.props.tags[`${typeTagHandle}`].map((item) => {
        return item.tag;
      });
    }
    // Select>options
    if (tagDetail !== null) {
      options = tagDetail.map((item, index) => {
        return (
          <option value={item} key={index}>
            {item}
          </option>
        );
      });
    }

    return (
      <form onSubmit={this.onSubmitHandler}>
        {/* Language */}
        <div className="statis-controller-wrp">
          <p className="statis-type">Language</p>
          <div className="statis-ctroller d-flex">
            <RadioButton
              label="English"
              onChange={this.handleChange("lang")}
              selected={lang === "en"}
              value="en"
              styleClass="mr-3"
            />
            <RadioButton
              label="Vietnamese"
              onChange={this.handleChange("lang")}
              selected={lang === "vn"}
              value="vn"
            />
          </div>
        </div>

        {/* Numbers */}
        <div className="statis-controller-wrp">
          <p className="statis-type">Numbers</p>
          <div className="statis-ctroller">
            {/* All word */}
            <RadioButton
              label="All words"
              onChange={this.handleChange("num")}
              selected={num === "all"}
              value="all"
              styleClass="mb-2"
            />
            {/* End all word */}

            {/* Top word */}
            <div className="d-flex align-items-center">
              <RadioButton
                label="Top"
                onChange={this.handleChange("num")}
                selected={num === "top"}
                value="top"
              />
              <input
                type="number"
                name="count"
                value={this.state.count}
                onChange={(e) =>
                  this.setState({
                    [e.target.name]: e.target.value,
                  })
                }
                disabled={num === "top" ? false : true}
                className="ml-2"
              />
            </div>
            {/*End top word  */}
          </div>
        </div>

        {/* Tag */}
        <div className="statis-controller-wrp">
          <p className="statis-type">Tag</p>
          <div className="statis-ctroller d-flex">
            <div className="mr-2">
              <RadioButton
                label="No tag"
                onChange={this.handleChange("typeTag")}
                selected={typeTag === ""}
                value=""
                styleClass="mb-2"
              />
              <RadioButton
                label="Pos"
                onChange={this.handleChange("typeTag")}
                selected={typeTag === "pos"}
                value="pos"
                styleClass="mb-2"
              />
              <RadioButton
                label="Ner"
                onChange={this.handleChange("typeTag")}
                selected={typeTag === "ner"}
                value="ner"
              />
            </div>
            <div className="tag__content">
              <select
                className="tag__select"
                name="typeTagDetail"
                onChange={(e) =>
                  this.setState({
                    [e.target.name]: e.target.value,
                  })
                }
                value={typeTagDetail}
                disabled={typeTag !== "" ? false : true}
              >
                <option defaultValue="Choose your option">
                  Choose your option
                </option>
                {options}
              </select>
            </div>
          </div>
        </div>

        {/* Button */}
        <div className="statis-button">
          <button type="submit" className="btn-search  mr-3">
            FILTER
          </button>
          <button
            type="button"
            className="btn-refresh"
            onClick={this.refreshHandler}
          >
            REFRESH
          </button>
        </div>
      </form>
    );
  }
}

const mapStateToProps = (state) => {
  return {
    tags: state.Tag.tags,
  };
};

export default connect(mapStateToProps)(StatisController);
