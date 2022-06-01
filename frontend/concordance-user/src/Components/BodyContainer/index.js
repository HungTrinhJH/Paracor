import React from "react";
import Sidebar from "../../Layouts/Sidebar";
const BodyContainer = (props) => {
  return (
    <div className="grid">
      <div className="grid-left">
        <Sidebar data={props.data}></Sidebar>
      </div>
      <div className="grid-right ml-2">{props.children}</div>
    </div>
  );
};

export default BodyContainer;
