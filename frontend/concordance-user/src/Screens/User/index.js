import React, { Component } from "react";
import UserController from "./UserController";
import Alert from "../../Layouts/Alert";
import Linebreak from "../../Components/Linebreak";
export default class User extends Component {
  render() {
    return (
      <div class="myContainer">
        <UserController />
        <Linebreak />
        <Alert />
      </div>
    );
  }
}
