import React, { Component } from "react";
import ShowLanguage from "../../../Components/Language";
import Linebreak from "../../../Components/Linebreak";
import Table from "../../../Components/Table";
import { connect } from "react-redux";
import "./Search.css";
import SearchController from "./SearchController";
import Modal from "../../../Components/Modal";
import Backdrop from "../../../Components/Backdrop";
import { ArcherContainer, ArcherElement } from "react-archer";
import $ from "jquery";
import { SEARCH_TYPE } from "../../../Redux/Action/type";
import Alert from "../../../Layouts/Alert";
import { Pagination } from "antd";
import { createAction } from "../../../Redux/Action";

class Search extends Component {
  state = {
    source: [],
    target: [],
    total: 0,
    currentPage: 1,
    rowSelected: null,
    modalToggle: false,
    searchType: "Match",
    language: "english",
  };
  handleRowSelected = (id) => {
    this.setState({
      rowSelected: id,
    });
  };
  handleSearchType = (type) => {
    this.setState({
      ...this.state,
      searchType: type,
    });
  };
  openModalHandler = () => {
    this.setState({
      modalToggle: !this.state.modalToggle,
    });
  };
  resetCurrentPage = (page) =>
    this.setState({ ...this.state, currentPage: page });

  componentDidMount() {
    $(".sourceIDSearch, .targetIDSearch").on("scroll", function () {
      $(".sourceIDSearch, .targetIDSearch").scrollTop($(this).scrollTop());
    });
  }

  onPageChangeHandler = (pageNumber) => {
    this.setState({
      ...this.state,
      currentPage: pageNumber,
    });
  };

  languageChangeHandler = (language) => {
    this.setState({
      ...this.state,
      language,
    });
  };
 
  refreshHandler = () => {
    this.props.dispatch(createAction(SEARCH_TYPE, "Match"));
    this.setState({
      total: 0,
      source: [],
      target: [],
      rowSelected: null,
      language: "english",
      currentPage: 1,
      searchType: "Match",
    });
  };
  getSearchData = (data) => {
    const { source, target, total } = data;
    this.setState({
      ...this.state,
      source,
      target,
      total,
    });
  };
  render() {
    // Data for table
    let { source, target, total, language, currentPage } = this.state;

    let sourceData = source;
    let targetData = target;

    let renderSource = null,
      renderTarget = null;
    if (this.props.detailSentence !== null) {
      renderSource = this.props.detailSentence.source.map((item, index) => {
        let linkArr = item[6].trim().split(",");
        if (linkArr[0] !== "-") {
          let relationsResult = [];
          for (let i = 0; i < linkArr.length; i++) {
            let obj = {
              targetId: "target" + linkArr[i],
              targetAnchor: "top",
              sourceAnchor: "bottom",
              style: { strokeColor: "#275efe", strokeWidth: 1 },
            };
            relationsResult.push(obj);
          }
          return (
            <ArcherElement id={item[3]} relations={relationsResult} key={index}>
              <div className={"word-item"}>
                <p className="mb-2">{item[8]}</p>
                <p className="mb-2">{item[4]}</p>
              </div>
            </ArcherElement>
          );
        } else
          return (
            <div className={"word-item"}>
              <p className="mb-2">{item[8]}</p>
              <p className="mb-2">{item[4]}</p>
            </div>
          );
      });
      renderTarget = this.props.detailSentence.target.map((item, index) => {
        let idRes = item[3];
        if (idRes.slice(0, 1) === "0") idRes = idRes.slice(1);
        idRes = "target" + idRes;
        return (
          <ArcherElement id={idRes} key={index}>
            <div className={"word-item"}>
              <p>{item[4]}</p>
              <p className="mt-1">{item[8]}</p>
            </div>
          </ArcherElement>
        );
      });
    }

    return (
      <div className="myContainer">
        <div className="row pb-2">
          <ShowLanguage
            languageChangeHandler={this.languageChangeHandler}
            language={language}
          />
          <SearchController
            handleSearchType={this.handleSearchType}
            handleRowSelected={this.handleRowSelected}
            getSearchData={this.getSearchData}
            refreshHandler={this.refreshHandler}
            language={language}
            currentPage={currentPage}
            resetCurrentPage={this.resetCurrentPage}
          />
        </div>
        <Linebreak />
        <Alert />
        <div className="d-flex justify-content-between align-items-center">
          <p className="search-result">
            Found total: <span>{total}</span> results , Type:
            <span> {this.state.searchType}</span>
          </p>

          <Pagination
            showQuickJumper
            pageSize={100}
            total={total}
            defaultCurrent={1}
            current={currentPage}
            showSizeChanger={false}
            onChange={this.onPageChangeHandler}
            responsive={true}
          />
        </div>

        {/* Search table */}
        <Table
          languageTitle="Source language"
          key="source"
          data={sourceData}
          handleRowSelected={this.handleRowSelected}
          selectedID={this.state.rowSelected}
          openModalHandler={this.openModalHandler}
          classID="sourceIDSearch"
        />
        <Table
          languageTitle="Target language"
          key="target"
          data={targetData}
          handleRowSelected={this.handleRowSelected}
          selectedID={this.state.rowSelected}
          openModalHandler={this.openModalHandler}
          classID="targetIDSearch"
        />
        {/* Modal show alignment */}
        <Modal
          show={this.state.modalToggle}
          modalClosed={this.openModalHandler}
        >
          <div className="myModal-top">
            <p className="modal-title">Show Alignment</p>
            <i className="fa fa-times" onClick={this.openModalHandler}></i>
          </div>
          <div className="myModal-content">
            <div className="modal-content__left">
              <div className="modal-content__item-left mt-2">
                <p className="mb-2">POS</p>
                <p className="source-title">Source</p>
              </div>
              <div className="modal-content__item-left">
                <p className="source-title">Target</p>
                <p className="mt-2 mb-3">POS</p>
              </div>
            </div>
            <div className="modal-content__right">
              <div className="line-source"></div>
              <ArcherContainer>
                <div className="modal-content__item-right d-flex source justify-content-around align-items-center fs-14">
                  {renderSource}
                </div>
                <div className="modal-content__item-right d-flex justify-content-around align-items-center fs-14">
                  {renderTarget}
                </div>
              </ArcherContainer>
            </div>
          </div>
        </Modal>
        <Backdrop
          show={this.state.modalToggle}
          clicked={this.openModalHandler}
        />
      </div>
    );
  }
}

// Get data from store
const mapStateToProps = (state) => {
  return {
    detailSentence: state.Data.detailSentence,
    type: state.SearchType.type,
  };
};
export default connect(mapStateToProps)(Search);
