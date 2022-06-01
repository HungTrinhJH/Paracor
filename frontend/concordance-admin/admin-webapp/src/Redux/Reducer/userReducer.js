import { FETCH_CREDENTIALS, LOG_OUT } from "../Action/type";

const initialState = {
  credentials: null,
  isAuthenticated: false,
};

const UserReducer = (state = initialState, action) => {
  switch (action.type) {
    case FETCH_CREDENTIALS: {
      let cloneState = { ...state };
      cloneState.credentials = action.payload;
      cloneState.isAuthenticated = true;
      state = cloneState;
      return { ...state };
    }
    case LOG_OUT: {
      let cloneState = { ...state };
      cloneState.credentials = action.payload;
      cloneState.isAuthenticated = false;
      state = cloneState;
      localStorage.removeItem("credentials")
      return { ...state };
    }
    default:
      return state;
  }
};

export default UserReducer;
