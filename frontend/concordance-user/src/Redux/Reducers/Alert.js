import { SET_ALERT } from "../Action/type";

const initialState = {
  msg: null,
};

const AlertReducer = (state = initialState, action) => {
  switch (action.type) {
    case SET_ALERT:
      state.msg = action.payload;
      return { ...state };

    default:
      return { ...state };
  }
};

export default AlertReducer;
