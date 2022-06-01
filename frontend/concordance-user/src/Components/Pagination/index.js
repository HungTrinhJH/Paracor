import React, { Component } from "react";
import "./pagination.css";
import { connect } from "react-redux";
import { createAction } from "../../Redux/Action";

class Pagination extends Component {
  state = {
    pageNumber: this.props.pageNumber,
  };
  handlePage = (value) => {
    let currentPageNumber = parseInt(this.props.pageNumber, 10);
    let newPageNumber;
    if (value === -1) {
      currentPageNumber > 1
        ? (newPageNumber = currentPageNumber + value)
        : (newPageNumber = 1);
    } else newPageNumber = currentPageNumber + value;

    this.setState(
      {
        pageNumber: newPageNumber,
      },
      () => {
        this.props.dispatch(
          createAction(this.props.typeSearch, this.state.pageNumber)
        );
      }
    );
  };
  handleOnChange = (e) => {
    this.setState({ pageNumber: e.target.value }, () => {
      this.props.dispatch(
        createAction(this.props.typeSearch, this.state.pageNumber)
      );
    });
  };
  handleOnBlur = (e) => {
    let { name, value } = e.target;
    let re = /^\d+$/;

    if (!re.test(value) || -value < 1) {
      this.setState(
        {
          [name]: 1,
        },
        () => {
          this.props.dispatch(
            createAction(this.props.typeSearch, this.state.pageNumber)
          );
        }
      );
    }
  };

  clickNextHandler = () => {
    if (this.props.nextPage) {
      return this.handlePage(1);
    }
    if (this.props.totalPage === 0) {
      return null;
    }
    if (this.state.pageNumber >= this.props.totalPage) return null;
    return this.handlePage(1);
  };
  clickPreviousHandler = () => {
    if (this.state.pageNumber === 1) {
      return null;
    }

    return this.handlePage(-1);
  };
  render() {
    return (
      <div className="pagination">
        <div className="d-flex justify-content-around align-items-center">
          <i
            className="fa fa-chevron-circle-left"
            onClick={this.clickPreviousHandler}
          ></i>
          <input
            type="number/text"
            value={this.props.pageNumber}
            className="pagination__number"
            name="pageNumber"
            onChange={this.handleOnChange}
            onBlur={this.handleOnBlur}
          />
          <i
            className="fa fa-chevron-circle-right"
            onClick={this.clickNextHandler}
          ></i>
        </div>
      </div>
    );
  }
}

export default connect()(Pagination);
