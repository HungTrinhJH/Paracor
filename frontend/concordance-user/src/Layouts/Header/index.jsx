import React from "react";
import "./Header.css";
import { Link, NavLink } from "react-router-dom";
import dataService from "../../Services/DataServices";
export default function Header() {
  return (
    <header className="header">
      <nav className="navbar navbar-expand-lg  fixed-top ">
        {/* Header Logo */}
        <Link className="navbar-brand logo d-flex align-items-center" to="/">
          <img src="/images/logo1.jpg" className="logo-img mr-3" alt="logo" />
          <span className="website_name">PARACOR</span>
        </Link>
        {/* End header logo */}

        {/* Header button toggler */}
        <button
          className="navbar-toggler"
          type="button"
          data-toggle="collapse"
          data-target="#navbarSupportedContent"
          aria-controls="navbarSupportedContent"
          aria-expanded="false"
          aria-label="Toggle navigation"
        >
          <span className="navbar-toggler-icon" />
        </button>
        {/* End button toggler */}

        {/* Navbar item */}
        <div className="collapse navbar-collapse" id="navbarSupportedContent">
          <ul className="navbar-nav ml-auto navItems">
            <li className="nav-item">
              <NavLink
                className="nav-link"
                exact={true}
                activeClassName="is-active"
                to="/"
              >
                Home
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink
                className="nav-link"
                exact={true}
                activeClassName="is-active"
                to="/user"
              >
                User
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink
                className="nav-link"
                exact={true}
                activeClassName="is-active"
                to="/corpus"
              >
                Corpus
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink
                className="nav-link"
                to="/faq"
                activeClassName="is-active"
              >
                FAQ
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink
                className="nav-link"
                to="/about-us"
                activeClassName="is-active"
              >
                About Us
              </NavLink>
            </li>
          </ul>
        </div>
        {/* End navbar item */}

        <div className="direct-admin">
          <i className="fa fa-user-edit"></i>
          <a
            href={dataService.adminURL}
            target="_blank"
            rel="noopener noreferrer"
            data-toggle="tooltip"
            data-placement="left"
            title="Go to admin"
            className="admin-icon"
          >
            <i className="fa fa-user-md" aria-hidden="true"></i>
          </a>
        </div>
      </nav>
    </header>
  );
}
