import React from "react";
import './interface.css'
import ChatBox from './chatBox'
class MessageContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {messages:[]}
    }

    render() {
        return (
            <ChatBox >
                {this.props.messages}
            </ChatBox>
        );
    }


}

export default MessageContainer;