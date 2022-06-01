import { combineReducers } from "redux";
import SearchTypeReducer from "./SearchTypeReducer";
import TagReducer from "./TagReducer";
import ControllerReducer from "./ControllerReducer";
import DataReducer from "./DataReducer";
import AlertReducer from "./Alert";

const RootReducer = combineReducers({
  SearchType: SearchTypeReducer,
  Tag: TagReducer,
  Controller: ControllerReducer,
  Data: DataReducer,
  Alert: AlertReducer,
});

export default RootReducer;
