import React, { Component } from "react";
import { connect } from "react-redux";
import TableHome from "./TableHome";
import { Pagination } from "antd";
import { UPDATE_PAGE_HOME } from "../../Redux/Action/type";
import Modal from "../../Components/Modal";
import Backdrop from "../../Components/Backdrop";
import { ArcherContainer, ArcherElement } from "react-archer";
import $ from "jquery";
import { createAction } from "../../Redux/Action";

class ParallelSentence extends Component {
  state = {
    rowSelected: null,
    modalToggle: false,
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
  onPageChangeHandler = (pageNumber) => {
    this.props.dispatch(createAction(UPDATE_PAGE_HOME, pageNumber));
  };

  render() {
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
            <ArcherElement id={item[3]} relations={relationsResult} key={index}>
              <div className={"word-item"} >
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
          <ArcherElement id={idRes} key={index}>
            <div className={"word-item"} >
              <p>{item[4]}</p>
              <p className="mt-1">{item[8]}</p>
            </div>
          </ArcherElement>
        );
      });
    }

    return (
      <div className="home myContainer pt-4">
        <div className="home__controller mb-3">
          <Pagination
            showQuickJumper
            pageSize={10}
            total={this.props.totalPage}
            defaultCurrent={1}
            current={this.props.pageNumber}
            showSizeChanger={false}
            onChange={this.onPageChangeHandler}
            responsive={true}
          />
        </div>
        <div className="home__table">
          <TableHome
            data={this.props.enData}
            selectedID={this.state.rowSelected}
            openModalHandler={this.openModalHandler}
            handleRowSelected={this.handleRowSelected}
            classID="sourceIDHome"
          />
          <TableHome
            data={this.props.viData}
            selectedID={this.state.rowSelected}
            openModalHandler={this.openModalHandler}
            handleRowSelected={this.handleRowSelected}
            classID="targetIDHome"
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
    );
  }
}

const mapStateToProps = (state) => {
  return {
    languageType: state.Controller.language,
    pageNumber: state.Controller.currentPageHome,
    viData: state.Data.initData.viData,
    enData: state.Data.initData.enData,
    detailSentence: state.Data.detailSentence,
    totalPage: state.Data.totalPageHome,
  };
};

export default connect(mapStateToProps)(ParallelSentence);
