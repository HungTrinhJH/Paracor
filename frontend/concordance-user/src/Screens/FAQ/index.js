import React, { Component } from "react";
import Term from "../Term";
import Help from "../Help";

export default class FAQ extends Component {
  render() {
    return (
      <>
        <div>
          <ul className="nav nav-tabs bg-nav-tab" id="myTab" role="tablist">
            <li className="nav-item">
              <a
                className="nav-link active"
                id="home-tab"
                data-toggle="tab"
                href="#home"
                role="tab"
                aria-controls="home"
                aria-selected="true"
              >
                Term
              </a>
            </li>
            <li className="nav-item">
              <a
                className="nav-link"
                id="profile-tab"
                data-toggle="tab"
                href="#profile"
                role="tab"
                aria-controls="profile"
                aria-selected="false"
              >
                User Manual
              </a>
            </li>
          </ul>
          <div className="tab-content" id="myTabContent">
            <div
              className="tab-pane fade show active container"
              id="home"
              role="tabpanel"
              aria-labelledby="home-tab"
            >
              <Term />
            </div>
            <div
              className="tab-pane fade container"
              id="profile"
              role="tabpanel"
              aria-labelledby="profile-tab"
            >
              <Help />
            </div>
          </div>
        </div>
      </>
    );
  }
}
