import React from "react";
import './interface.css'
import Message from "./Message";
import InitQuestion from "./InitQuestion";

class ChatBox extends React.Component {
    constructor(props) {
        super(props);
        this.inp = React.createRef();
        this.state= {
            messages: [],
            script :"",
            inputValue:""
        }
        this.handleBtn = this.handleBtn.bind(this);
        this.updateInputValue = this.updateInputValue.bind(this);

    }

    handleBtn(event){
        this.setState({messages:[...this.state.messages,<Message cls={"mr"} value={this.state.inputValue}/>]})
        console.log(this.state.messages)
    }
    updateInputValue(event) {
        console.log(event.target.value)
        this.setState({
            inputValue:event.target.value
        })
    }


    render() {
        const getScript = () => {
            return this.state.script;
        }
        return (
            <div>
                <div className="tbl">

                </div>
                <div>
                    <div className="container-fluid box" id="chat">
                        <InitQuestion/>
                        {this.state.messages.map((item,i)=>{
                            console.log("adding item",item)
                            return item
                        })}
                    </div>
                    <div className="inp">
                        <input className="inp-txt" type="text" id="inpTxt" onChange={evt => this.updateInputValue(evt)}/>
                        <input type="button" className="inp-btn" id="inpBtn" onClick={this.handleBtn}/>
                    </div>
                </div>
            </div>
        )
    }
}
export default ChatBox