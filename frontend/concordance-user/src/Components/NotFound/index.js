import React from "react";
import "./NotFound.css";

import { Link } from "react-router-dom";
const NotFound = () => (
  <section className="page_404">
    <div className="wrp_404">
      <h1 className="text-center">404 Not Found</h1>
      <div className="d-flex justify-content-center">
        <Link to="/" className="link_404">
          Back to Homepage
        </Link>
      </div>
      <div className="page_img">
        <img src="/images/dribbble_1.gif" alt="not found img" />
      </div>
    </div>
  </section>
);

export default NotFound;
