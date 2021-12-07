import React from "react";
import './interface.css'
import MessageContainer from "./MessageContainer";


class Message extends React.Component {
    constructor(props) {
        super(props);
    }

    render(){
        return(
            <div className={this.props.cls}> {this.props.value}</div>
        )

    }
}

export default Message;