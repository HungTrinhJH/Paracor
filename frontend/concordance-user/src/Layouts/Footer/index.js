import React from "react";
import "./footer.css";
const Footer = () => {
  return (
    <footer className="footer">
      <div className="my-container">
        <div className="footer__top d-flex align-items-center">
          <p>Paracor Website</p>
        </div>
        <div className="footer__bottom mt-2 ">
          <div className="d-flex justify-content-between">
            <div>
              <p className="footer__title">Contact</p>
              <div className="footer__item">
                <ul>
                  <li className="mb-1">
                    <a
                      href="https://goo.gl/maps/Nwq3GxXnMWLLHNgZ7"
                      style={{ color: "white" }}
                      className="mb-1"
                    >
                      Room C44, Building C, 227 Nguyen Van Cu, District 5, Ho
                      Chi Minh City, Vietnam.
                    </a>
                    <div>
                      <iframe
                        src="https://www.google.com/maps/embed?pb=!1m14!1m8!1m3!1d7839.261546387637!2d106.682172!3d10.762913!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x0%3A0x43900f1d4539a3d!2sVietnam%20National%20University%20Ho%20Chi%20Minh%20City%20-%20University%20of%20Science!5e0!3m2!1sen!2sus!4v1606063536194!5m2!1sen!2sus"
                        width="250"
                        height="150"
                        frameBorder="0"
                        style={{ border: "0" }}
                        aria-hidden="false"
                        tabIndex="0"
                        title="map"
                      ></iframe>
                    </div>
                  </li>
                  <li>Phone number: (028) 66 849 856</li>
                  <li>Email: clc@hcmus.edu.vn</li>
                </ul>
              </div>
            </div>
            <div>
              <div>
                <p className="footer__title">About us</p>
                <div className="footer__item">
                  <div>
                    <h5 className="text-white">Developer</h5>
                    <ul>
                      <li>Trịnh Vũ Minh Hùng (minhhung.it.work@gmail.com)</li>
                      <li>Lê Hoài Bảo (lehoaibao081999@gmail.com)</li>
                    </ul>
                  </div>
                  <div>
                    <h5 className="text-white">Supervisor</h5>
                    <ul>
                      <li>Hoàng Khuê</li>
                      <li>Lương An Vinh</li>
                      <li>Đinh Điền</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
            <div>
              <p className="footer__title">Partner</p>
              <div className="footer__item">
                <ul>
                  <li>COMPUTATIONAL LINGUISTICS CENTER</li>
                  <li>
                    <a
                      href="http://www.clc.hcmus.edu.vn/"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <img
                        src="/images/cropped-Logo-v2.0.22.png"
                        alt="logo-clc"
                      />
                    </a>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
        <div className="footer__license">
          <p>
            Copyright(C){" "}
            <a
              href="http://www.clc.hcmus.edu.vn/"
              target="_blank"
              rel="noopener noreferrer"
            >
              CLC
            </a>{" "}
            2020 Version 0.1
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
