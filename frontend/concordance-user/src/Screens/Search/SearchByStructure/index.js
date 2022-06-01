import React, { Component } from "react";
import ShowLanguage from "../../../Components/Language";
import Linebreak from "../../../Components/Linebreak";
import { Pagination } from "antd";
import { connect } from "react-redux";
import TableHome from "../../ParallelSentence/TableHome";
import $ from "jquery";
import Modal from "../../../Components/Modal";
import Backdrop from "../../../Components/Backdrop";
import { ArcherContainer, ArcherElement } from "react-archer";
import SearchStructureController from "./SearchByStructureController";
import Alert from "../../../Layouts/Alert";
class SearchByStructure extends Component {
  state = {
    source: [],
    target: [],
    language: "english",
    total: 0,
    currentPage: 1,
    rowSelected: null,
    modalToggle: false,
  };
  shouldComponentUpdate(nextProps, nextState) {
    if (this.props.data !== nextProps.data) {
      this.setState({
        data: nextProps.data,
      });
    }
    return true;
  }
  resetCurrentPage = (page) => {
    this.setState({
      ...this.state,
      currentPage: page,
    });
  };
  componentDidMount() {
    $(".sourceIDHome, .targetIDHome").on("scroll", function () {
      $(".sourceIDHome, .targetIDHome").scrollTop($(this).scrollTop());
    });
  }
  handleRowSelected = (id) => {
    this.setState({
      rowSelected: id,
    });
  };

  openModalHandler = () => {
    this.setState({
      modalToggle: !this.state.modalToggle,
    });
  };

  getAllData = (data) => {
    let dataResult = { source: [], target: [], total: 0 };
    let source = [];
    let target = [];
    if (data) {
      source = Object.values(data.source)[0];
      target = Object.values(data.target)[0];

      for (let i = 0; i < source.length; i++) {
        dataResult["source"].push({
          sentence_id: source[i][0],
          sentence: source[i][2],
        });
        dataResult["target"].push({
          sentence_id: target[i][0],
          sentence: target[i][1],
        });
      }
      dataResult.total = data.total;
    }
    this.setState({
      ...this.state,
      source: dataResult.source,
      target: dataResult.target,
      total: dataResult.total,
    });
  };

  languageChangeHandler = (language) => {
    this.setState({
      ...this.state,
      language,
    });
  };

  onPageChangeHandler = (pageNumber) => {
    this.setState({
      ...this.state,
      currentPage: pageNumber,
    });
  };
  refreshHandler = () => {
    this.setState({
      total: 0,
      source: [],
      target: [],
      rowSelected: null,
      language: "english",
      currentPage: 1,
    });
  };
  render() {
    let { source, target, total, currentPage, language } = this.state;

    let sourceData = source;
    let targetData = target;

    let renderSource = null,
      renderTarget = null;
    if (this.props.detailSentence !== null) {
      renderSource = this.props.detailSentence.source.map((item, index) => {
        let linkArr = item[6].trim().split(",");
        if (item[6] !== "0") {
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
            <ArcherElement id={item[3]} relations={relationsResult}>
              <div className={"word-item"} key={index}>
                <p className="mb-2">{item[8]}</p>
                <p className="mb-2">{item[4]}</p>
              </div>
            </ArcherElement>
          );
        } else
          return (
            <div className={"word-item"} key={index}>
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
          <ArcherElement id={idRes}>
            <div className={"word-item"} key={index}>
              <p>{item[4]}</p>
              <p className="mt-1">{item[8]}</p>
            </div>
          </ArcherElement>
        );
      });
    }

    return (
      <div>
        <div className="myContainer">
          {/* Search Controller */}
          <div className="row pb-2">
            <ShowLanguage
              languageChangeHandler={this.languageChangeHandler}
              language={language}
            />
            <SearchStructureController
              getAllData={this.getAllData}
              currentPage={currentPage}
              language={language}
              handleRowSelected={this.handleRowSelected}
              refreshHandler={this.refreshHandler}
              resetCurrentPage={this.resetCurrentPage}
            />
          </div>
          <Linebreak />
          <Alert />
          {/* Paginatipon */}
          <div className="home__controller my-2">
            <div className="d-flex align-items-center justify-content-between">
              <p className="search-result">
                Found total: <span>{total} </span>results
              </p>
              <Pagination
                showQuickJumper
                pageSize={100}
                total={total}
                currentPage={1}
                current={this.state.currentPage}
                showSizeChanger={false}
                onChange={this.onPageChangeHandler}
                responsive={true}
              />
            </div>
          </div>

          <div className="home__table">
            <TableHome
              data={sourceData}
              openModalHandler={this.openModalHandler}
              handleRowSelected={this.handleRowSelected}
              classID="sourceIDHome"
              selectedID={this.state.rowSelected}
            />
            <TableHome
              data={targetData}
              openModalHandler={this.openModalHandler}
              handleRowSelected={this.handleRowSelected}
              classID="targetIDHome"
              selectedID={this.state.rowSelected}
            />
          </div>
          {/* Modal WordAlignment */}
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
      </div>
    );
  }
}

const mapStateToProps = (state) => {
  return {
    detailSentence: state.Data.detailSentence,
  };
};

export default connect(mapStateToProps)(SearchByStructure);
