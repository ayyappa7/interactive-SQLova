import React from 'react';
import MessageContainer from "./MessageContainer";

class Welcome extends React.Component {
    render() {
        return (
            <div>
                <h1>MISP</h1>
                <MessageContainer/>

            </div>
        );
    }
}

export default Welcome;